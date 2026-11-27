"""The admission and settlement Lua scripts (PLAN.md §7, ADR-0005).

Every key a request touches shares the hash tag ``{t:<team id>}``, so both scripts work
on Redis Cluster too. Times are Redis ``TIME`` in milliseconds (one clock for every
replica), unless a test passes its own.

Admission, in one round trip:

1. Budget counters must exist; if one does not (a new month, or Redis lost it), the
   script returns ``rebuild`` and the caller seeds it from the Postgres ledger.
2. Expired reservations (a replica died mid-request) are released.
3. For each budget with a hard limit: ``spent + reserved + estimate ≤ limit``, else
   ``budget``.
4. Token buckets (RPM and TPM, per key and per team) refill at ``capacity / 60 s``. If
   any lacks the request's cost, nothing is taken from any of them and the reply is
   ``rate`` with the wait for the slowest one. A cost above a bucket's capacity needs a
   full bucket, so one huge request is never blocked forever.
5. Otherwise every bucket is charged and every budget reserves the estimate.

Settlement moves a reservation into ``spent``, corrects the TPM buckets by actual minus
estimated tokens (debt is bounded at one full bucket), and reports a soft limit crossed
for the first time in the period (``SET NX``).
"""

ADMIT = """
local now
if ARGV[1] ~= '' then
  now = tonumber(ARGV[1])
else
  local t = redis.call('TIME')
  now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
end
local nb = tonumber(ARGV[2])
local nd = tonumber(ARGV[3])
local member = ARGV[4]
local life = tonumber(ARGV[5])
local spend_ttl = tonumber(ARGV[6])
local bucket_ttl = tonumber(ARGV[7])
local base = 8

local missing = {}
for j = 1, nd do
  if redis.call('EXISTS', KEYS[nb + 2 * j - 1]) == 0 then missing[#missing + 1] = j end
end
if #missing > 0 then return {'rebuild', unpack(missing)} end

for j = 1, nd do
  local spend, resv = KEYS[nb + 2 * j - 1], KEYS[nb + 2 * j]
  local expired = redis.call('ZRANGEBYSCORE', resv, '-inf', now)
  if #expired > 0 then
    local released = 0
    for _, m in ipairs(expired) do
      released = released + tonumber(string.match(m, '|(%d+)$'))
    end
    redis.call('HINCRBY', spend, 'reserved', -released)
    redis.call('ZREMRANGEBYSCORE', resv, '-inf', now)
  end
  local hard = tonumber(ARGV[base + 2 * nb + 2 * (j - 1)])
  local estimate = tonumber(ARGV[base + 2 * nb + 2 * (j - 1) + 1])
  if hard >= 0 then
    local v = redis.call('HMGET', spend, 'spent', 'reserved')
    local spent = tonumber(v[1]) or 0
    local reserved = tonumber(v[2]) or 0
    if spent + reserved + estimate > hard then
      return {'budget', j, spent, reserved}
    end
  end
end

local levels = {}
local blocked, wait = 0, 0
for i = 1, nb do
  local cap = tonumber(ARGV[base + 2 * (i - 1)])
  local cost = tonumber(ARGV[base + 2 * (i - 1) + 1])
  local rate = cap / 60000
  local v = redis.call('HMGET', KEYS[i], 'tokens', 'ts')
  local level, ts = tonumber(v[1]), tonumber(v[2])
  if level == nil then level, ts = cap, now end
  level = math.min(cap, level + math.max(0, now - ts) * rate)
  levels[i] = level
  local need = math.min(cost, cap)
  if level < need then
    local w = math.ceil((need - level) / rate)
    if w > wait then wait, blocked = w, i end
  end
end
if blocked > 0 then
  local out = {'rate', blocked, wait}
  for i = 1, nb do out[#out + 1] = math.floor(levels[i]) end
  return out
end

local out = {'ok', now}
for i = 1, nb do
  local cap = tonumber(ARGV[base + 2 * (i - 1)])
  local cost = tonumber(ARGV[base + 2 * (i - 1) + 1])
  local level = levels[i] - math.min(cost, cap)
  redis.call('HSET', KEYS[i], 'tokens', tostring(level), 'ts', now)
  redis.call('PEXPIRE', KEYS[i], bucket_ttl)
  out[#out + 1] = math.floor(level)
end
for j = 1, nd do
  local spend, resv = KEYS[nb + 2 * j - 1], KEYS[nb + 2 * j]
  local estimate = tonumber(ARGV[base + 2 * nb + 2 * (j - 1) + 1])
  redis.call('HINCRBY', spend, 'reserved', estimate)
  redis.call('ZADD', resv, now + life, member .. '|' .. estimate)
  redis.call('EXPIRE', resv, spend_ttl)
end
return out
"""

SETTLE = """
local nb = tonumber(ARGV[1])
local nd = tonumber(ARGV[2])
local member = ARGV[3]
local alert_ttl = tonumber(ARGV[4])
local base = 5

for i = 1, nb do
  local cap = tonumber(ARGV[base + 2 * (i - 1)])
  local correction = tonumber(ARGV[base + 2 * (i - 1) + 1])
  local level = tonumber(redis.call('HGET', KEYS[i], 'tokens'))
  if level ~= nil and correction ~= 0 then
    level = math.max(-cap, math.min(cap, level - correction))
    redis.call('HSET', KEYS[i], 'tokens', tostring(level))
  end
end

local out = {}
for j = 1, nd do
  local spend = KEYS[nb + 3 * j - 2]
  local resv = KEYS[nb + 3 * j - 1]
  local alert = KEYS[nb + 3 * j]
  local actual = tonumber(ARGV[base + 2 * nb + 3 * (j - 1)])
  local estimate = tonumber(ARGV[base + 2 * nb + 3 * (j - 1) + 1])
  local soft = tonumber(ARGV[base + 2 * nb + 3 * (j - 1) + 2])
  local spent, fired = -1, 0
  if redis.call('ZREM', resv, member .. '|' .. estimate) == 1 then
    redis.call('HINCRBY', spend, 'reserved', -estimate)
  end
  -- A counter that vanished (Redis lost it) is not recreated here: the next admission
  -- rebuilds it from the ledger, which will include this request.
  if redis.call('EXISTS', spend) == 1 then
    spent = redis.call('HINCRBY', spend, 'spent', actual)
    if soft >= 0 and spent >= soft then
      if redis.call('SET', alert, '1', 'NX', 'EX', alert_ttl) then fired = 1 end
    end
  end
  out[#out + 1] = spent
  out[#out + 1] = fired
end
return out
"""

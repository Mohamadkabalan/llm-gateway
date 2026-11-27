# k6 with the xk6-sse extension, built from pinned versions so every bench run uses the
# same load generator. (k6 v2 can also fetch extensions on demand, but then the binary
# would depend on when the run happened.)
#
# k6 is pinned to v2.0.0 on purpose: xk6-sse v0.2.0 is tested against it, and under
# k6 v2.4.0 the SSE client object reaches scripts without its methods (`client.on` is
# undefined). Checked on 2026-11-27; revisit when xk6-sse tags a newer release.
FROM grafana/xk6:1.5.1 AS build
RUN xk6 build v2.0.0 \
      --with github.com/phymbert/xk6-sse@v0.2.0 \
      --output /tmp/k6

FROM grafana/k6:2.4.0
COPY --from=build /tmp/k6 /usr/bin/k6

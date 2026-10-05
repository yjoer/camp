FROM ubuntu:26.04

RUN apt-get update \
	&& apt-get install -y --no-install-recommends ca-certificates curl \
	&& install -m 0755 -d /usr/share/postgresql-common/pgdg \
	&& curl --fail -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc https://www.postgresql.org/media/keys/ACCC4CF8.asc \
	&& echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt resolute-pgdg main" > /etc/apt/sources.list.d/pgdg.list \
	&& apt-get update \
	&& apt-get install -y --no-install-recommends pgbouncer=1.26.0-4.pgdg26.04+1 \
	&& rm -rf /var/lib/apt/lists/* \
	&& groupadd --gid 1001 pgbouncer \
	&& useradd --gid 1001 --no-create-home --system --shell /usr/sbin/nologin --uid 1001 pgbouncer

USER 1001:1001

EXPOSE 6432

ENTRYPOINT ["pgbouncer", "/etc/pgbouncer/pgbouncer.ini"]

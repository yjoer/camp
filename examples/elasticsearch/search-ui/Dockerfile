FROM node:23 AS builder

WORKDIR /app
COPY . .

RUN npm install -g pnpm@latest-10
RUN pnpm install

RUN pnpm build

FROM node:23-slim

WORKDIR /app
COPY --from=builder /app/dist dist

RUN npm install -g http-server

EXPOSE 8080

CMD ["http-server", "dist"]

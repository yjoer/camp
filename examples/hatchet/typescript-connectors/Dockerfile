FROM node:23-slim

WORKDIR /app
COPY . .

RUN corepack enable
RUN pnpm install --prod

CMD ["pnpm", "run", "start:worker"]

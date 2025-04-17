FROM node:23-slim

WORKDIR /app
COPY . .

RUN npm install -g pnpm@latest-10
RUN pnpm install --prod

CMD ["pnpm", "start:worker"]

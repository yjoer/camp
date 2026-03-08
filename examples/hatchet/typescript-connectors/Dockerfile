FROM node:23-slim

WORKDIR /app
COPY . .

RUN corepack enable
RUN yarn workspaces focus --production

CMD ["yarn", "start:worker"]

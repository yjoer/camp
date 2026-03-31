FROM node:23-slim

WORKDIR /app
COPY . .

RUN corepack enable && yarn set version stable
RUN yarn workspaces focus --production

CMD ["yarn", "start:worker"]

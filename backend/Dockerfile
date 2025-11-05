FROM node:22-slim

WORKDIR /app

COPY package*.json ./
RUN npm install --production

COPY . .

RUN mkdir -p /app/uploads

EXPOSE 5050
CMD ["node", "server.js"]
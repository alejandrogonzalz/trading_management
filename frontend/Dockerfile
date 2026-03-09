FROM node:20-slim

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# Environment variable for the backend API
ENV VITE_API_URL=http://localhost:8000

CMD ["npm", "run", "dev", "--", "--host"]

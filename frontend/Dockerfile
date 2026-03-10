FROM node:20-slim

WORKDIR /app

COPY package*.json ./
RUN rm -rf node_modules package-lock.json # Aggressively clear old node_modules and lock file
RUN npm install # Use npm install for robustness

COPY . .

# Environment variable for the backend API
ENV VITE_API_URL=http://localhost:8001

CMD ["npm", "run", "dev", "--", "--host"]

/* Simple static dev server for this repo.
   Usage:
     npm install
     npm run dev
   Then open: http://localhost:5173/
*/
const express = require('express');
const compression = require('compression');
const morgan = require('morgan');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 5173;

app.use(compression());
app.use(morgan('dev'));

// Helpful CORS for local dev
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  next();
});

// Serve everything from repo root
const rootDir = path.resolve(__dirname, '..'); // one level up is repo root
app.use(express.static(rootDir, { extensions: ['html'], index: 'index.html' }));

// Ensure root serves index.html
app.get('/', (req, res) => {
  res.sendFile(path.join(rootDir, 'index.html'));
});

app.listen(PORT, () => {
  console.log(`Dev server running at http://localhost:${PORT}`);
});

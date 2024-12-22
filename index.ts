import express, { type Request, type Response } from "express";

const app = express();
const port = process.env["PORT"] || 3001;

app.get("/", (_: Request, res: Response) => {
  res.send("Hello World");
});

app.listen(port, () => {
  console.log(`Listening on port ${port}..`);
});

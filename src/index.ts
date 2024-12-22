import express, { type Express, type Request, type Response } from "express";
import dotenv from "dotenv";

dotenv.config();

const app: Express = express();
const port = process.env["PORT"] || 3000;

app.use(express.json());

// Basic route
app.get("/", (req: Request, res: Response) => {
    res.json({ message: "Welcome to the Express API" });
});

// Health check endpoint
app.get("/health", (req: Request, res: Response) => {
    res.json({ status: "OK" });
});

app.listen(port, () => {
    console.log(`⚡️[server]: Server is running at http://localhost:${port}`);
});

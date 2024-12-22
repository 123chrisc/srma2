import crypto from "crypto";

class Task {}

class Batch {
    private readonly id: string;
    private readonly createdAt: Date;

    private tasks: Task[];
    constructor() {
        this.id = crypto.randomUUID().slice(-8);
        this.createdAt = new Date();
        this.tasks = [];
    }
}

export default Batch;

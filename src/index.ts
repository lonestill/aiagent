import { BrowserAgent } from "./agent.js";
import * as readline from "readline";
import "dotenv/config";

// Persistent readline interface for user input prompts
let rl: readline.Interface | null = null;

function getUserInput(prompt: string): Promise<string> {
  if (!rl) {
    rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });
  }

  return new Promise((resolve, reject) => {
    rl!.question(`✏️ ${prompt}: `, (answer) => {
      resolve(answer.trim());
    });
    
    rl!.on('close', () => {
      reject(new Error("Input stream closed"));
    });
  });
}

function cleanup() {
  if (rl) {
    rl.close();
    rl = null;
  }
}

process.on('SIGINT', () => {
  cleanup();
  process.exit(0);
});

process.on('exit', cleanup);

async function main() {
  const task = process.argv.slice(2).join(" ");

  if (!task) {
    console.error("❌ Нужно указать задачу. Пример: npm run dev 'закажи пиццу'");
    process.exit(1);
  }

  if (!process.env.OPENAI_APIKEY) {
    console.error("❌ OPENAI_APIKEY не найден в .env файле");
    process.exit(1);
  }

  const agent = new BrowserAgent({
    model: process.env.OPENAI_MODEL || "gpt-4o",
    maxSteps: 100,
    headless: false,
  });

  try {
    await agent.run(task, getUserInput);
    console.log("\n✅ Задача выполнена!");
  } catch (error) {
    console.error(`\n❌ Ошибка агента: ${error}`);
    if (error instanceof Error && error.stack) {
      console.error(error.stack);
    }
    process.exit(1);
  } finally {
    cleanup();
  }
}

main();

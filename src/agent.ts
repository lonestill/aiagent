import OpenAI from "openai";
import { chromium, Page, BrowserContext } from "playwright";
import path from "path";
import { z } from "zod";
import "dotenv/config";
import * as fs from "fs";

export type AgentConfig = {
  model: string;
  maxSteps: number;
  headless: boolean;
};

type ToolCall = OpenAI.Chat.Completions.ChatCompletionMessageToolCall;

type Observation = {
  url: string;
  title: string;
  elements: Array<{
    element_id: number;
    role: string;
    name: string;
    value: string;
  }>;
  headings: string[];
};

type UserProfile = {
  identity: {
    name: string;
    phone: string;
    email: string;
  };
  locations: {
    home: {
      address: string;
      city: string;
      apartment: string;
      floor: string;
      entrance: string;
      intercom: string;
    };
    work: {
      address: string;
      city: string;
    };
  };
  preferences: {
    food: {
      dislikes: string[];
      likes: string[];
      dietary_restrictions: string[];
      budget_max: number;
    };
    delivery: {
      prefer_free_delivery: boolean;
      default_tip_percent: number;
    };
  };
  payment: {
    preferred_method: string;
    saved_cards: any[];
  };
};

type ToolOutcome = {
  ok: boolean;
  detail: string;
};

const defaultConfig: AgentConfig = {
  model: process.env.OPENAI_MODEL ?? "gpt-4o",
  maxSteps: 100,
  headless: (process.env.HEADLESS ?? "false").toLowerCase() === "true",
};

// Единый селектор для сбора элементов (используется и в snapshot, и в executeTool)
const INTERACTIVE_SELECTOR = 'a, button, input, textarea, select, [role="button"], [role="combobox"], [onclick], [tabindex], [data-id], [data-product], [data-item], div[class*="card"], div[class*="item"], div[class*="product"]';

// Use existing Chrome installation + profile to persist cookies/sessions.
const chromeExecutable =
  process.env.CHROME_EXECUTABLE ?? "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const chromeProfileDir =
  process.env.CHROME_PROFILE_DIR ?? "C:\\Users\\silas\\AppData\\Local\\Google\\Chrome\\User Data\\Default";

// Optional: attach to already-running Chrome via remote debugging.
const remoteDebugUrl =
  process.env.CHROME_REMOTE_DEBUG_URL ||
  (process.env.CHROME_REMOTE_DEBUG_PORT ? `http://localhost:${process.env.CHROME_REMOTE_DEBUG_PORT}` : undefined);

const toolSchemas: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "open_url",
      description: "Open a URL in the active tab.",
      parameters: {
        type: "object",
        properties: {
          url: { type: "string", description: "Absolute URL to navigate to." },
        },
        required: ["url"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "click",
      description:
        "Click on an element by its index number. Look at the INTERACTIVE ELEMENTS list and provide the index [N] of the element you want to click.",
      parameters: {
        type: "object",
        properties: {
          index: { type: "number", description: "Index number of the element to click (e.g., 5 for element [5])." },
        },
        required: ["index"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "fill",
      description: "Fill text into an input field by its index.",
      parameters: {
        type: "object",
        properties: {
          index: { type: "number", description: "Index number of the input field to fill." },
          text: { type: "string", description: "Text to type." },
          pressEnter: { type: "boolean", description: "Press Enter after typing." },
        },
        required: ["index", "text"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "press",
      description: "Press a keyboard key (e.g., Enter, Tab, ArrowDown).",
      parameters: {
        type: "object",
        properties: {
          key: { type: "string", description: "Key name for page.keyboard.press" },
        },
        required: ["key"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "scroll",
      description: "Scroll the page by a delta in pixels.",
      parameters: {
        type: "object",
        properties: {
          dx: { type: "number", description: "Horizontal delta." },
          dy: { type: "number", description: "Vertical delta." },
        },
        required: ["dy"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "wait_for_navigation",
      description: "Wait for the next navigation or network idle to settle the page.",
      parameters: {
        type: "object",
        properties: {
          timeoutMs: { type: "number", description: "Timeout in milliseconds." },
        },
        required: [],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "go_back",
      description: "Go back to the previous page in browser history.",
      parameters: {
        type: "object",
        properties: {},
      },
    },
  },
];

const snapshotLimit = {
  textNodes: 120,
  interactive: 100,
};

export class BrowserAgent {
  private readonly openai: OpenAI;
  private readonly config: AgentConfig;
  private readonly userProfile: UserProfile;

  constructor(config?: Partial<AgentConfig>) {
    this.config = { ...defaultConfig, ...config };
    this.openai = new OpenAI({ apiKey: process.env.OPENAI_APIKEY });
    
    // Загружаем User Profile согласно ТЗ
    const profilePath = path.join(process.cwd(), 'user-profile.json');
    try {
      const profileData = fs.readFileSync(profilePath, 'utf-8');
      this.userProfile = JSON.parse(profileData);
      console.log("✓ User Profile загружен");
    } catch (error) {
      console.log("⚠️  User Profile не найден, используем пустой");
      this.userProfile = {
        identity: { name: "", phone: "", email: "" },
        locations: {
          home: { address: "", city: "", apartment: "", floor: "", entrance: "", intercom: "" },
          work: { address: "", city: "" }
        },
        preferences: {
          food: { dislikes: [], likes: [], dietary_restrictions: [], budget_max: 2000 },
          delivery: { prefer_free_delivery: true, default_tip_percent: 0 }
        },
        payment: { preferred_method: "card", saved_cards: [] }
      };
    }
  }

  async run(task: string, getUserInput?: (prompt: string) => Promise<string>): Promise<void> {
    let context: BrowserContext;
    let page: Page;
    let shouldCloseContext = true;

    if (remoteDebugUrl) {
      const browser = await chromium.connectOverCDP(remoteDebugUrl);
      context = browser.contexts()[0] ?? (await browser.newContext({ viewport: { width: 1400, height: 900 } }));
      page = await context.newPage();
      shouldCloseContext = false; // do not close user's running browser
    } else {
      context = await chromium.launchPersistentContext(chromeProfileDir, {
        executablePath: chromeExecutable,
        headless: this.config.headless,
        args: ["--disable-blink-features=AutomationControlled"],
        viewport: { width: 1400, height: 900 },
      });
      page = context.pages()[0] ?? (await context.newPage());
    }
    
    // Auto-dismiss dialogs (alerts, confirms, prompts)
    page.on('dialog', async (dialog) => {
      console.log(`⚠️  Диалог: ${dialog.type()} - "${dialog.message()}"`);
      await dialog.accept();
      console.log(`  → Автоматически принял диалог`);
    });
    
    await page.goto("about:blank");

    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: this.buildSystemPrompt(),
      },
      {
        role: "user",
        content: `Goal: ${task}`,
      },
    ];

    console.log("\n🤖 Агент запущен. Задача:", task);
    console.log("🌐 Браузер открыт, собираю информацию о странице...\n");
    
    const initialSnapshot = await capturePageSnapshot(page);
    messages.push({ role: "user", content: renderObservation(initialSnapshot, this.userProfile) });
    
    let consecutiveScrolls = 0;
    let recentActions: { tool: string; params: string }[] = [];
    const maxRecentActions = 5;

    for (let step = 1; step <= this.config.maxSteps; step++) {
      console.log(`\n📍 Шаг ${step}/${this.config.maxSteps}: Думаю что делать дальше...`);
      
      // Check for repeated identical actions
      const actionRepeatCount = recentActions.filter(
        (a) => a.tool === 'click' && recentActions[recentActions.length - 1]?.params === a.params
      ).length;
      
      if (actionRepeatCount >= 3) {
        console.log(
          "⚠️  ВНИМАНИЕ: Кликаю на одно и то же место уже " + actionRepeatCount + " раз. Нужно попробовать другой подход!"
        );
        messages.push({
          role: "user",
          content: `⚠️ IMPORTANT: You clicked element [${recentActions[recentActions.length - 1]?.params}] ${actionRepeatCount} times. The page didn't change. Try scrolling down or clicking on a different element.`,
        });
      }
      
      const completion = await this.openai.chat.completions.create({
        model: this.config.model,
        messages,
        tools: toolSchemas,
        tool_choice: "auto",
        temperature: 0.2,
      });

      const choice = completion.choices[0]?.message;
      if (!choice) {
        console.log("❌ Нет ответа от модели");
        messages.push({ role: "assistant", content: "No response; stopping." });
        break;
      }

      if (choice.tool_calls && choice.tool_calls.length > 0) {
        console.log(`🔧 Выполняю действия: ${choice.tool_calls.map(c => c.function?.name).join(", ")}`);
        
        // First add the assistant message with tool calls
        messages.push({
          role: "assistant",
          content: choice.content || "",
          tool_calls: choice.tool_calls,
        });
        
        // Then add all tool results
        for (const call of choice.tool_calls as ToolCall[]) {
          const outcome = await this.executeTool(call, page);
          
          // Track action for repeat detection
          const toolParams = call.function.arguments;
          if (call.function.name === 'click') {
            recentActions.push({ tool: 'click', params: toolParams });
            if (recentActions.length > maxRecentActions) recentActions.shift();
          }
          
          console.log(`  → ${call.function.name}: ${outcome.ok ? "✓" : "✗"} ${outcome.detail.substring(0, 100)}`);
          messages.push({ role: "tool", tool_call_id: call.id, content: outcome.detail });
          
          // Track consecutive scrolls to prevent infinite loops
          if (call.function.name === "scroll") {
            consecutiveScrolls++;
          } else {
            consecutiveScrolls = 0;
            recentActions = []; // Reset action history on non-scroll action
          }
        }
        
        // Prevent infinite scrolling
        if (consecutiveScrolls > 10) {
          console.log("⚠️  Слишком много скроллов подряд, прерываю...");
          messages.push({ role: "user", content: "You've scrolled too many times consecutively. Please try a different approach or complete the task with available information." });
          consecutiveScrolls = 0;
        }
        
        // Get fresh snapshot after all tools have been executed
        console.log("📸 Собираю новую информацию о странице...");
        
        // Check if page is still valid
        if (page.isClosed()) {
          console.log("❌ Страница закрылась, останавливаю агента.");
          throw new Error("Page was closed during tool execution");
        }
        
        // Wait a bit for any navigation to complete
        await page.waitForLoadState('load', { timeout: 3000 }).catch(() => {});
        
        const snap = await capturePageSnapshot(page);
        messages.push({ role: "user", content: renderObservation(snap, this.userProfile) });
        continue;
      }

      // Model decided to respond directly (done or needs info)
      if (choice.content) {
        console.log(`\n💬 Агент:\n${choice.content}\n`);

        const lowerContent = choice.content.toLowerCase();

        // Only ask user for truly critical data: credentials, payment, or one-time verification codes
        const needsLoginOrPayment = getUserInput && (
          lowerContent.includes('логин') ||
          lowerContent.includes('пароль') ||
          lowerContent.includes('login') ||
          lowerContent.includes('password') ||
          lowerContent.includes('оплат') ||
          lowerContent.includes('payment') ||
          lowerContent.includes('телефон') ||
          lowerContent.includes('номер') ||
          lowerContent.includes('phone')
        );

        const needsVerificationCode = getUserInput && (
          lowerContent.includes('код') ||
          lowerContent.includes('sms') ||
          lowerContent.includes('смс') ||
          lowerContent.includes('otp') ||
          lowerContent.includes('однораз') ||
          lowerContent.includes('2fa') ||
          lowerContent.includes('двухфактор') ||
          lowerContent.includes('push') ||
          lowerContent.includes('пуш') ||
          lowerContent.includes('подтвержд') ||
          lowerContent.includes('verification')
        );

        const needsUserResponse = needsLoginOrPayment || needsVerificationCode;

        if (needsUserResponse) {
          const promptText = needsVerificationCode ? "\n📲 Введите код из СМС/приложения: " : "\n👤 Введите данные: ";
          const userResponse = await getUserInput(promptText);
          
          if (userResponse && userResponse.trim()) {
            messages.push({ role: "assistant", content: choice.content });
            messages.push({ role: "user", content: userResponse });
            console.log(`\n🔄 Продолжаю работу...\n`);
            continue;
          }
        }

        // Fallback: always give the user a chance to add an instruction instead of exiting.
        if (getUserInput) {
          const extraInput = await getUserInput("\n✏️ Доп. инструкция (Enter — завершить): ");
          if (extraInput && extraInput.trim()) {
            messages.push({ role: "assistant", content: choice.content });
            messages.push({ role: "user", content: extraInput });
            console.log(`\n🔄 Продолжаю работу...\n`);
            continue;
          }
        }
      }
      break;
    }

    try {
      if (shouldCloseContext) {
        await context.close();
      }
    } catch (err) {
      console.error("Error closing browser:", err);
    }
  }

  private async executeTool(call: ToolCall, page: Page): Promise<ToolOutcome> {
    const toolName = call.function.name;
    const toolArgs = call.function.arguments;
    
    try {
      switch (toolName) {
        case "open_url": {
          const { url } = z.object({ url: z.string().url() }).parse(JSON.parse(toolArgs));

          // Prevent navigating to known broken login endpoints
          if (/samokat\.ru\/login/i.test(url)) {
            return { ok: false, detail: `Blocked navigation to ${url}; use on-page login popup or checkout flow instead.` };
          }

          await page.goto(url, { waitUntil: "load", timeout: 15000 });
          await page.waitForTimeout(2500);
          return { ok: true, detail: `Navigated to ${url}.` };
        }
        case "click": {
          const { index } = z.object({ index: z.number() }).parse(JSON.parse(toolArgs));
          try {
            const elements = await page.locator('a, button, input, textarea, select, [role="button"], [role="combobox"], [onclick], [tabindex], [data-id], [data-product], [data-item], div[class*="card"], div[class*="item"], div[class*="product"]').all();
            
            if (index < 0 || index >= elements.length) {
              return { ok: false, detail: `Invalid index ${index}. Available elements: 0-${elements.length - 1}` };
            }
            
            const element = elements[index];
            
            // Check if element would open new tab/window or navigate away
            const href = await element.getAttribute('href').catch(() => null);
            const target = await element.getAttribute('target').catch(() => null);
            const currentDomain = new URL(page.url()).hostname;
            
            if (target === '_blank') {
              return { ok: false, detail: `Element [${index}] has target="_blank" (new tab). Skipping.` };
            }
            
            if (href) {
              try {
                const linkUrl = new URL(href, page.url());
                const linkDomain = linkUrl.hostname;
                
                // Skip external links (different domain)
                if (linkDomain !== currentDomain && !href.startsWith('#') && !href.startsWith('/')) {
                  return { ok: false, detail: `Element [${index}] leads to external site ${linkDomain}. Skipping.` };
                }
              } catch (e) {
                // Invalid URL, probably relative or anchor - OK to click
              }
            }
            
            await element.scrollIntoViewIfNeeded({ timeout: 5000 }).catch(() => {});
            await page.waitForTimeout(200);
            
            // Use Promise.race to handle navigation vs. no navigation
            await Promise.race([
              element.click({ timeout: 3000 }),
              page.waitForTimeout(3500)
            ]);
            
            await page.waitForTimeout(500);
            return { ok: true, detail: `Clicked element [${index}].` };
          } catch (error) {
            const msg = error instanceof Error ? error.message : String(error);
            if (msg.includes('closed') || msg.includes('detached')) {
              return { ok: false, detail: `Element [${index}] triggered page close/navigation.` };
            }
            return { ok: false, detail: `Failed to click element [${index}]: ${msg}` };
          }
        }
        case "fill": {
          const { index, text, pressEnter } = z
            .object({ index: z.number(), text: z.string(), pressEnter: z.boolean().optional() })
            .parse(JSON.parse(toolArgs));
          try {
            const elements = await page.locator('input, textarea').all();
            
            if (index < 0 || index >= elements.length) {
              return { ok: false, detail: `Invalid index ${index}. Available inputs: 0-${elements.length - 1}` };
            }
            
            const element = elements[index];
            await element.scrollIntoViewIfNeeded();
            await page.waitForTimeout(200);
            await element.fill(text, { timeout: 4000 });
            if (pressEnter) {
              await element.press("Enter");
            }
            return { ok: true, detail: `Filled element [${index}] with text.` };
          } catch (error) {
            throw new Error(`Failed to fill element [${index}]: ${error}`);
          }
        }
        case "press": {
          const { key } = z.object({ key: z.string() }).parse(JSON.parse(toolArgs));
          await page.keyboard.press(key);
          return { ok: true, detail: `Pressed ${key}.` };
        }
        case "scroll": {
          const { dx, dy } = z
            .object({ dx: z.number().default(0), dy: z.number() })
            .parse(JSON.parse(toolArgs));
          await page.mouse.wheel(dx ?? 0, dy);
          return { ok: true, detail: `Scrolled by dx=${dx ?? 0}, dy=${dy}.` };
        }
        case "wait_for_navigation": {
          const { timeoutMs } = z.object({ timeoutMs: z.number().optional() }).parse(JSON.parse(toolArgs));
          await page.waitForLoadState("load", { timeout: timeoutMs ?? 10000 });
          return { ok: true, detail: "Navigation wait completed." };
        }
        case "go_back": {
          await page.goBack();
          return { ok: true, detail: "Navigated back." };
        }
        default:
          return { ok: false, detail: `Unknown tool ${toolName}.` };
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return { ok: false, detail: `Tool ${toolName} failed: ${message}` };
    }
  }

  private buildSystemPrompt(): string {
    return [
      "You are Shadow-User, an autonomous web-automation agent. Your task: execute user commands with MINIMAL human interaction.",
      "",
      "=== CORE PRINCIPLE: AUTONOMY ===",
      "**First check memory (User Profile), then act. Only interrupt for critical missing data.**",
      "",
      "USER PROFILE contains:",
      "- identity (phone, email, name)",
      "- locations (home address, work address)",  
      "- preferences (food likes/dislikes, budget)",
      "- payment (preferred method)",
      "",
      "=== INTERRUPT RULES (когда спрашивать пользователя) ===",
      "ASK ONLY for:",
      "1. **Missing Secret Data**: Passwords, CVV, 2FA codes, SMS codes",
      "2. **Critical Action Confirmation**: Final payment button (\"Оплатить\", \"Pay\")",
      "3. **Missing Required Data**: If field is required and not in User Profile",
      "",
      "DO NOT ASK for:",
      "- Data already in User Profile (address, phone, email)",
      "- Navigation decisions (which category to click, which item to choose)",
      "- Trivial confirmations (cookies, age verification - handle automatically)",
      "",
      "=== ELEMENT INTERACTION ===",
      "Elements are provided in JSON format (Semantic Tree from Accessibility API):",
      "{\"element_id\": N, \"role\": \"button\", \"name\": \"Add to cart\"}",
      "",
      "To interact: click({index: N}) where N is element_id.",
      "",
      "AVAILABLE TOOLS:",
      "- open_url: Navigate to URL",
      "- click: Click element by element_id",
      "- fill: Fill input field by element_id",
      "- press: Press keyboard key",
      "- scroll: Scroll page (use sparingly)",
      "- wait_for_navigation: Wait for page load",
      "- go_back: Browser back button",
      "",
      "=== WORKFLOW FOR FOOD ORDERING ===",
      "1. Check if logged in (look for account/profile elements)",
      "2. If not logged in & phone not in Profile: ASK user for phone",
      "3. Navigate to food category (use headings and buttons)",
      "4. Find specific item by name (search in headings and element names)",
      "5. Add to cart",
      "6. Go to checkout",
      "7. Fill address from User Profile (home.address + city)",
      "8. Before final payment button: STOP and ask confirmation",
      "",
      "=== CRITICAL RULES ===",
      "- Use User Profile data FIRST, don't ask if data exists",
      "- Read PAGE HEADINGS to find categories and item names",
      "- Match element_id from JSON list to click",
      "- If item not visible: scroll down and get fresh snapshot",
      "- Stop ONLY at payment confirmation or missing secrets",
      "",
      "REMEMBER: You are AUTONOMOUS. Make smart decisions. Check Profile before asking!",
      "",
    ].join("\n");
  }
}

async function capturePageSnapshot(page: Page) {
  try {
    await page.waitForLoadState('domcontentloaded', { timeout: 2000 }).catch(() => {});
    
    const url = page.url();
    const title = await page.title();
    
    // Собираем заголовки для контекста (названия товаров, категорий)
    const headings = await page.evaluate(() => {
      const h = Array.from(document.querySelectorAll('h1, h2, h3, h4'));
      return h.map(el => el.textContent?.trim()).filter(t => t && t.length > 2).slice(0, 30);
    });
    
    // Собираем ТОЛЬКО видимые интерактивные элементы в viewport (для экономии токенов)
    const elements = await page.evaluate(() => {
      const selector = 'a, button, input, textarea, select, [role="button"], [role="combobox"], [onclick], [tabindex], [data-id], [data-product], [data-item], div[class*="card"], div[class*="item"], div[class*="product"]';
      const allElements = Array.from(document.querySelectorAll(selector));
      
      const viewportHeight = window.innerHeight;
      const viewportWidth = window.innerWidth;
      
      const filtered = allElements
        .map((el, idx) => {
          const rect = el.getBoundingClientRect();
          const style = window.getComputedStyle(el);
          
          // Проверяем видимость
          const isVisible = 
            rect.width > 0 &&
            rect.height > 0 &&
            style.display !== 'none' &&
            style.visibility !== 'hidden' &&
            style.opacity !== '0';
          
          // Проверяем нахождение в viewport или близко к нему (с запасом 500px)
          const inViewport = 
            rect.top < viewportHeight + 500 &&
            rect.bottom > -100 &&
            rect.left < viewportWidth &&
            rect.right > 0;
          
          if (!isVisible || !inViewport) return null;
          
          // Получаем текст элемента
          let text = el.textContent?.trim() || '';
          if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) {
            text = el.placeholder || el.value || el.name || '';
          }
          
          const role = el.getAttribute('role') || el.tagName.toLowerCase();
          
          return {
            element_id: idx,
            role: role,
            name: text.slice(0, 80),
            value: el instanceof HTMLInputElement ? (el.value || '') : ''
          };
        })
        .filter((el): el is { element_id: number; role: string; name: string; value: string } => 
          el !== null && (el.name.length > 0 || el.role === 'button' || el.role === 'link')
        )
        .slice(0, 50); // Ограничиваем 50 элементами для экономии токенов
      
      return filtered;
    });
    
    return { url, title, elements, headings };
  } catch (error) {
    console.log("⚠️  Ошибка при сборе snapshot, повторяю...");
    await page.waitForLoadState('load', { timeout: 5000 }).catch(() => {});
    const url = page.url();
    const title = await page.title();
    return { url, title, elements: [], headings: [] };
  }
}

function renderObservation(obs: Observation, userProfile: UserProfile): string {
  const lines: string[] = [];
  lines.push(`URL: ${obs.url}`);
  lines.push(`Title: ${obs.title}`);
  lines.push("");
  
  // Показываем заголовки (названия категорий, товаров)
  if (obs.headings.length) {
    lines.push("=== PAGE HEADINGS ===");
    for (const heading of obs.headings) {
      lines.push(`  ${heading}`);
    }
    lines.push("");
  }
  
  // Показываем доступные данные из User Profile
  lines.push("=== YOUR AVAILABLE DATA ===");
  if (userProfile.identity.phone) lines.push(`Phone: ${userProfile.identity.phone}`);
  if (userProfile.identity.email) lines.push(`Email: ${userProfile.identity.email}`);
  if (userProfile.locations.home.address) lines.push(`Home: ${userProfile.locations.home.address}, ${userProfile.locations.home.city}`);
  if (userProfile.preferences.food.budget_max) lines.push(`Budget: up to ${userProfile.preferences.food.budget_max}₽`);
  lines.push("");
  
  // Показываем элементы в формате JSON согласно ТЗ ("No-HTML" approach)
  if (obs.elements.length) {
    lines.push("=== INTERACTIVE ELEMENTS (Semantic Tree) ===");
    lines.push("Use element_id to click: click({index: N})");
    lines.push("");
    
    for (const el of obs.elements) {
      // Формат JSON согласно ТЗ
      const jsonEl = {
        element_id: el.element_id,
        role: el.role,
        name: el.name,
        ...(el.value && { value: el.value })
      };
      lines.push(JSON.stringify(jsonEl));
    }
  } else {
    lines.push("No interactive elements found.");
  }
  
  return lines.join("\n");
}

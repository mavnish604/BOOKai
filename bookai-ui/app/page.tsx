"use client";

import React, { useEffect, useRef, useState } from 'react';
import Header from './components/Header';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import { BotIcon, SparkleIcon } from './components/Icons';

const API_URL = process.env.NEXT_PUBLIC_BACKEND_URL;
const QUERY_LIMIT = 5;
const SESSION_QUERY_COUNT_KEY = 'bookai-query-count';

type Message = {
  sender: 'bot' | 'user';
  text: string;
  recommendations: Recommendation[];
};

type Recommendation = {
  title: string;
  explanation: string;
  score: number;
};

const starterPrompts = [
  'A thoughtful mystery for the weekend',
  'Books like The Alchemist',
  'A short, uplifting read',
];

export default function Home() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    {
      sender: 'bot',
      text: 'Tell me what you are in the mood to read, and I’ll find your next great book.',
      recommendations: [],
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [queriesUsed, setQueriesUsed] = useState(0);
  const chatWindowRef = useRef<HTMLDivElement>(null);
  const isLimitReached = queriesUsed >= QUERY_LIMIT;

  useEffect(() => {
    const storedCount = Number.parseInt(window.sessionStorage.getItem(SESSION_QUERY_COUNT_KEY) ?? '0', 10);
    setQueriesUsed(Number.isFinite(storedCount) ? Math.min(Math.max(storedCount, 0), QUERY_LIMIT) : 0);
  }, []);

  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const submittedQuery = query.trim();
    if (!submittedQuery || isLoading || isLimitReached) return;

    const nextCount = queriesUsed + 1;
    setQueriesUsed(nextCount);
    window.sessionStorage.setItem(SESSION_QUERY_COUNT_KEY, String(nextCount));
    setIsLoading(true);
    setMessages((prev) => [...prev, { sender: 'user', text: submittedQuery, recommendations: [] }]);
    setQuery('');

    try {
      if (!API_URL) {
        throw new Error('NEXT_PUBLIC_BACKEND_URL is not configured');
      }

      const response = await fetch(`${API_URL}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: submittedQuery }),
      });

      if (!response.ok) {
        throw new Error(`Recommendation request failed with status ${response.status}`);
      }

      const data = await response.json();
      setMessages((prev) => [...prev, {
        sender: 'bot',
        text: data.response,
        recommendations: data.recommendations || [],
      }]);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      setMessages((prev) => [...prev, {
        sender: 'bot',
        text: 'I couldn’t reach the recommendation service just now. Please try again in a moment.',
        recommendations: [],
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-[100dvh] flex-col overflow-hidden bg-[#f8f9ff] text-slate-900 transition-colors dark:bg-[#090a12] dark:text-slate-100">
      <Header queriesUsed={queriesUsed} queryLimit={QUERY_LIMIT} />

      <main ref={chatWindowRef} className="relative flex-1 overflow-y-auto px-4 pb-6 pt-9 sm:px-6 sm:pt-14">
        <div className="pointer-events-none absolute inset-x-0 top-0 h-72 bg-[radial-gradient(ellipse_at_top,rgba(199,210,254,0.55),transparent_66%)] dark:bg-[radial-gradient(ellipse_at_top,rgba(67,56,202,0.18),transparent_66%)]" />
        <div className="relative mx-auto max-w-3xl">
          <section className="mb-10 text-center animate-fade-in-up">
            <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-indigo-100 bg-white/80 px-3 py-1.5 text-xs font-medium text-indigo-700 shadow-sm backdrop-blur dark:border-indigo-400/15 dark:bg-indigo-500/10 dark:text-indigo-300">
              <SparkleIcon className="h-3.5 w-3.5" />
              Powered by your curiosity
            </div>
            <h2 className="text-3xl font-semibold tracking-tight sm:text-5xl">
              Find a book you’ll <span className="bg-gradient-to-r from-indigo-600 via-violet-600 to-fuchsia-600 bg-clip-text text-transparent dark:from-indigo-300 dark:via-violet-300 dark:to-fuchsia-300">love.</span>
            </h2>
            <p className="mx-auto mt-4 max-w-lg text-sm leading-6 text-slate-500 sm:text-base dark:text-slate-400">
              Share a mood, a favourite author, or a feeling. BookAI will curate your next chapter.
            </p>
            {messages.length === 1 && (
              <div className="mt-7 flex flex-wrap justify-center gap-2">
                {starterPrompts.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    onClick={() => setQuery(prompt)}
                    disabled={isLimitReached}
                    className="rounded-full border border-slate-200 bg-white/80 px-3.5 py-2 text-xs text-slate-600 shadow-sm transition hover:-translate-y-0.5 hover:border-indigo-200 hover:text-indigo-700 hover:shadow-md disabled:cursor-not-allowed disabled:opacity-45 dark:border-white/10 dark:bg-white/5 dark:text-slate-300 dark:hover:border-indigo-400/30 dark:hover:text-indigo-200"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            )}
          </section>

          <section aria-label="Conversation">
            {messages.map((msg, index) => (
              <ChatMessage key={`${msg.sender}-${index}`} {...msg} />
            ))}

            {isLoading && (
              <div className="mb-7 flex animate-fade-in-up gap-3">
                <div className="mt-1 grid h-8 w-8 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 text-white shadow-md shadow-indigo-500/20">
                  <BotIcon className="h-4 w-4" />
                </div>
                <div className="flex items-center gap-1 rounded-2xl rounded-tl-sm bg-white px-4 py-3 shadow-sm ring-1 ring-slate-100 dark:bg-[#151622] dark:ring-white/7">
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-indigo-400 [animation-delay:-0.3s]" />
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-indigo-400 [animation-delay:-0.15s]" />
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-indigo-400" />
                </div>
              </div>
            )}
          </section>
        </div>
      </main>

      <ChatInput
        query={query}
        setQuery={setQuery}
        handleSubmit={handleSubmit}
        isLoading={isLoading}
        queriesUsed={queriesUsed}
        queryLimit={QUERY_LIMIT}
      />
    </div>
  );
}

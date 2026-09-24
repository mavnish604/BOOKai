"use client";

import React, { useEffect, useRef, useState } from 'react';
import Header from './components/Header';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import { BookAiMark, CompassIcon, SarvamMotif, SparklesIcon } from './components/Icons';

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

const showcaseAgents = [
    {
        id: 'mystery',
        label: 'Atmospheric Mystery',
        icon: '🕯️',
        desc: 'Richly crafted whodunits with psychological depth',
        prompt: 'A thoughtful, atmospheric mystery for a rainy weekend with deep character studies',
    },
    {
        id: 'philosophy',
        label: 'Soul & Philosophy',
        icon: '🧭',
        desc: 'Works exploring purpose, destiny, and personal mythologies',
        prompt: 'Books like The Alchemist and Siddhartha that explore self-discovery and destiny',
    },
    {
        id: 'uplifting',
        label: 'Short & Uplifting',
        icon: '☕',
        desc: 'Concise, heartwarming stories that leave an indelible glow',
        prompt: 'A short, uplifting novel under 200 pages that leaves you feeling inspired',
    },
    {
        id: 'scifi',
        label: 'Mind-Bending Sci-Fi',
        icon: '🌌',
        desc: 'Speculative fiction exploring consciousness, time, and cosmos',
        prompt: 'Hard sci-fi with philosophical weight exploring humanity, artificial intelligence, and time',
    },
];

const indicTopics = [
    'Indian Historical Epics',
    'Magical Realism',
    'Philosophical Fiction',
    'Thoughtful Memoirs',
    'Cosy Detective Tales',
    'Modern Sci-Fi Classics',
];

const initialBotMessage: Message = {
    sender: 'bot',
    text: 'Welcome to BookAI. Tell me what kind of theme, feeling, or narrative journey you are seeking, and I will curate your next read.',
    recommendations: [],
};

export default function Home() {
    const [query, setQuery] = useState('');
    const [messages, setMessages] = useState<Message[]>([initialBotMessage]);
    const [isLoading, setIsLoading] = useState(false);
    const [queriesUsed, setQueriesUsed] = useState(0);
    const [activeTab, setActiveTab] = useState('mystery');
    const chatWindowRef = useRef<HTMLDivElement>(null);
    const isLimitReached = queriesUsed >= QUERY_LIMIT;

    useEffect(() => {
        const storedCount = Number.parseInt(
            window.sessionStorage.getItem(SESSION_QUERY_COUNT_KEY) ?? '0',
            10
        );
        setQueriesUsed(Number.isFinite(storedCount) ? Math.min(Math.max(storedCount, 0), QUERY_LIMIT) : 0);
    }, []);

    useEffect(() => {
        if (chatWindowRef.current) {
            chatWindowRef.current.scrollTo({
                top: chatWindowRef.current.scrollHeight,
                behavior: 'smooth',
            });
        }
    }, [messages, isLoading]);

    const handleReset = () => {
        window.sessionStorage.removeItem(SESSION_QUERY_COUNT_KEY);
        setQueriesUsed(0);
        setMessages([initialBotMessage]);
        setQuery('');
    };

    const executeQuery = async (queryText: string) => {
        const submittedQuery = queryText.trim();
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
            setMessages((prev) => [
                ...prev,
                {
                    sender: 'bot',
                    text: data.response || 'Here are curated selections tailored to your reading intent:',
                    recommendations: data.recommendations || [],
                },
            ]);
        } catch (error) {
            console.error('Error fetching recommendations:', error);
            setMessages((prev) => [
                ...prev,
                {
                    sender: 'bot',
                    text: 'I could not connect to the recommendation service at this moment. Please check your connection or retry.',
                    recommendations: [],
                },
            ]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        await executeQuery(query);
    };

    return (
        <div className="flex h-[100dvh] flex-col overflow-hidden bg-[#f9fafe] text-[#1e2033] transition-colors dark:bg-[#0a0b12] dark:text-[#f3f4fa]">
            {/* Header */}
            <Header queriesUsed={queriesUsed} queryLimit={QUERY_LIMIT} onReset={handleReset} />

            {/* Main Interactive Stage */}
            <main ref={chatWindowRef} className="relative flex-1 overflow-y-auto px-4 pb-6 pt-8 sm:px-6 sm:pt-12">
                {/* Sarvam Signature Blue/Indigo Halo Glow */}
                <div
                    className="pointer-events-none absolute left-1/2 top-10 -z-10 h-72 w-72 md:h-[450px] md:w-[680px] -translate-x-1/2 opacity-35 blur-[90px] md:blur-[120px]"
                    style={{
                        background: 'radial-gradient(ellipse, #A5BBFC 0%, #D5E2FF 40%, transparent 70%)',
                    }}
                    aria-hidden="true"
                />

                <div className="relative mx-auto max-w-4xl">
                    {/* Sarvam AI-Inspired Hero Presentation */}
                    {messages.length === 1 && (
                        <section className="mb-10 text-center animate-fade-in-up">
                            {/* Sarvam Sacred Geometry Gateway Motif */}
                            <div className="flex justify-center text-indigo-600 dark:text-indigo-400">
                                <SarvamMotif className="h-7 md:h-9 w-auto" />
                            </div>

                            {/* Tagline Framed by Radial Gradient Dividers */}
                            <div className="mx-auto mt-4 flex w-fit flex-col items-center gap-1.5 md:gap-2">
                                <div className="sarvam-divider max-w-[280px]" />
                                <p className="font-matter text-xs font-medium uppercase tracking-[2px] text-[#2c3360] dark:text-[#a5b4fc]">
                                    India’s Intelligent Reading Platform
                                </p>
                                <div className="sarvam-divider max-w-[280px]" />
                            </div>

                            {/* Sarvam SeasonMix High-Contrast Display Headline */}
                            <h1 className="mt-5 font-season-mix text-4xl leading-[1.08] tracking-tight text-[#1e2033] sm:text-6xl dark:text-[#f3f4fa]">
                                Stories for all,{' '}
                                <span className="italic font-normal text-indigo-600 dark:text-indigo-400">
                                    curated by AI
                                </span>
                            </h1>

                            {/* Subtitle in Matter */}
                            <p className="mx-auto mt-3.5 max-w-xl font-matter text-sm leading-relaxed text-[#555973] sm:text-base dark:text-[#9da3be]">
                                Built on narrative intelligence. Curating books across traditions, philosophies, and moods with sovereign precision.
                            </p>

                            {/* Sarvam Dual Action Pill Buttons */}
                            <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
                                <button
                                    type="button"
                                    onClick={() => executeQuery(showcaseAgents[0].prompt)}
                                    disabled={isLimitReached}
                                    className="btn-sarvam-filled inline-flex items-center gap-2 rounded-full px-5 py-2.5 font-matter text-xs font-medium cursor-pointer shadow-sm disabled:opacity-40"
                                >
                                    <SparklesIcon className="h-3.5 w-3.5 text-indigo-200" />
                                    <span>Curate Atmospheric Mystery</span>
                                </button>
                                <button
                                    type="button"
                                    onClick={() => executeQuery(showcaseAgents[1].prompt)}
                                    disabled={isLimitReached}
                                    className="btn-sarvam-outline inline-flex items-center gap-2 rounded-full px-5 py-2.5 font-matter text-xs font-medium cursor-pointer disabled:opacity-40"
                                >
                                    <span>Explore Philosophical Reads</span>
                                </button>
                            </div>

                            {/* Sarvam Showcase Playground Frame (Tabs) */}
                            <div className="mt-10 overflow-hidden rounded-2xl md:rounded-3xl border border-black/10 bg-white/70 shadow-[0_16px_40px_rgba(30,32,51,0.06)] backdrop-blur-xl dark:border-white/10 dark:bg-[#12131e]/70">
                                {/* Segmented Tab Bar */}
                                <div
                                    className="flex items-center overflow-x-auto border-b border-[#e2e5f1] bg-[#f0f2f9]/70 p-1 dark:border-[#25283d] dark:bg-[#171827]/70"
                                    role="tablist"
                                    aria-label="Curation modes"
                                >
                                    {showcaseAgents.map((agent) => {
                                        const isSelected = activeTab === agent.id;
                                        return (
                                            <button
                                                key={agent.id}
                                                type="button"
                                                role="tab"
                                                aria-selected={isSelected}
                                                onClick={() => setActiveTab(agent.id)}
                                                className={`flex flex-1 min-w-[140px] items-center justify-center gap-2 rounded-xl py-2.5 px-3 font-matter text-xs font-medium transition-all duration-200 cursor-pointer ${
                                                    isSelected
                                                        ? 'bg-white text-[#1e2033] shadow-[0_2px_8px_rgba(30,32,51,0.08)] ring-1 ring-black/5 dark:bg-[#1f2135] dark:text-[#f3f4fa] dark:ring-white/10'
                                                        : 'text-[#555973] hover:text-[#1e2033] dark:text-[#8f94a8] dark:hover:text-[#f3f4fa]'
                                                }`}
                                            >
                                                <span>{agent.icon}</span>
                                                <span className="truncate">{agent.label}</span>
                                            </button>
                                        );
                                    })}
                                </div>

                                {/* Active Tab Interactive Preview */}
                                {showcaseAgents
                                    .filter((agent) => agent.id === activeTab)
                                    .map((agent) => (
                                        <div
                                            key={agent.id}
                                            className="dot-matrix relative flex flex-col items-center justify-between p-6 sm:p-8 text-center"
                                        >
                                            <div className="max-w-md">
                                                <p className="font-season-mix text-lg font-medium text-[#1e2033] dark:text-[#f3f4fa]">
                                                    {agent.label}
                                                </p>
                                                <p className="mt-1 font-matter text-xs text-[#555973] dark:text-[#9da3be]">
                                                    {agent.desc}
                                                </p>
                                                <div className="mt-4 rounded-xl border border-[#e2e5f1] bg-white/80 p-3 font-matter text-xs italic text-[#2c3360] dark:border-[#25283d] dark:bg-[#1a1b2b]/80 dark:text-[#c7d2fe]">
                                                    “{agent.prompt}”
                                                </div>
                                            </div>

                                            <button
                                                type="button"
                                                onClick={() => executeQuery(agent.prompt)}
                                                disabled={isLimitReached}
                                                className="btn-sarvam-filled mt-5 inline-flex items-center gap-2 rounded-full px-6 py-2.5 font-matter text-xs font-medium cursor-pointer"
                                            >
                                                <span>Ask BookAI This Prompt</span>
                                                <span>→</span>
                                            </button>
                                        </div>
                                    ))}
                            </div>

                            {/* Indic Topic Pills */}
                            <div className="mt-7 flex flex-wrap items-center justify-center gap-1.5">
                                <span className="inline-flex items-center gap-1 font-matter text-[11px] font-medium text-[#8f94a8] mr-1">
                                    <CompassIcon className="h-3 w-3" />
                                    Explore Themes:
                                </span>
                                {indicTopics.map((topic) => (
                                    <button
                                        key={topic}
                                        type="button"
                                        onClick={() => executeQuery(`Best books in ${topic}`)}
                                        disabled={isLimitReached}
                                        className="rounded-full border border-[#e2e5f1] bg-white px-3 py-1 font-matter text-[11px] text-[#555973] shadow-xs transition hover:border-[#c4c8db] hover:text-[#1e2033] dark:border-[#25283d] dark:bg-[#12131e] dark:text-[#9da3be] dark:hover:border-[#3d4263] dark:hover:text-[#f3f4fa] cursor-pointer"
                                    >
                                        {topic}
                                    </button>
                                ))}
                            </div>
                        </section>
                    )}

                    {/* Active Conversation Messages */}
                    <section aria-label="Conversation stream">
                        {messages.map((msg, index) => (
                            <ChatMessage key={`${msg.sender}-${index}`} {...msg} />
                        ))}

                        {/* Sarvam-grade Loading Indicator */}
                        {isLoading && (
                            <div className="mb-8 flex items-start gap-3.5 animate-fade-in-up">
                                <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl border border-[#e2e5f1] bg-white text-indigo-600 shadow-xs dark:border-[#25283d] dark:bg-[#171827] dark:text-indigo-400">
                                    <BookAiMark className="h-4.5 w-4.5 animate-pulse" />
                                </div>
                                <div className="flex items-center gap-2.5 rounded-2xl border border-[#e2e5f1] bg-white px-4 py-3 shadow-xs dark:border-[#25283d] dark:bg-[#12131e]">
                                    <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-indigo-600 [animation-delay:-0.3s]" />
                                    <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-indigo-600 [animation-delay:-0.15s]" />
                                    <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-indigo-600" />
                                    <span className="ml-1.5 font-matter text-xs text-[#555973] dark:text-[#9da3be]">
                                        Consulting sovereign book intelligence…
                                    </span>
                                </div>
                            </div>
                        )}
                    </section>
                </div>
            </main>

            {/* Chat Input Console Bar */}
            <ChatInput
                query={query}
                setQuery={setQuery}
                handleSubmit={handleSubmit}
                isLoading={isLoading}
                queriesUsed={queriesUsed}
                queryLimit={QUERY_LIMIT}
                onReset={handleReset}
            />
        </div>
    );
}

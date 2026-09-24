"use client";

import React, { useRef, useEffect } from 'react';
import { RefreshIcon, SendIcon, SparklesIcon } from './Icons';

interface ChatInputProps {
    query: string;
    setQuery: (query: string) => void;
    handleSubmit: (e: React.FormEvent) => void;
    isLoading: boolean;
    queriesUsed: number;
    queryLimit: number;
    onReset?: () => void;
}

export default function ChatInput({
    query,
    setQuery,
    handleSubmit,
    isLoading,
    queriesUsed,
    queryLimit,
    onReset,
}: ChatInputProps) {
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const isLimitReached = queriesUsed >= queryLimit;
    const remaining = Math.max(queryLimit - queriesUsed, 0);

    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
        }
    }, [query]);

    return (
        <footer className="relative flex-shrink-0 px-4 pb-4 pt-2 sm:px-6 sm:pb-6">
            {/* Ambient Background Gradient Fade */}
            <div className="pointer-events-none absolute inset-x-0 bottom-0 -z-10 h-32 bg-gradient-to-t from-[#f9fafe] via-[#f9fafe]/80 to-transparent dark:from-[#0a0b12] dark:via-[#0a0b12]/80" />

            <form className="mx-auto max-w-4xl" onSubmit={handleSubmit}>
                <div
                    className={`flex items-end gap-2.5 rounded-2xl border bg-white p-2.5 shadow-[0_8px_30px_rgba(30,32,51,0.08)] transition-all duration-200 dark:bg-[#12131e] dark:shadow-[0_8px_30px_rgba(0,0,0,0.35)] ${
                        isLimitReached
                            ? 'border-amber-300 bg-amber-50/20 dark:border-amber-700/40 dark:bg-amber-950/10'
                            : 'border-[#e2e5f1] focus-within:border-[#9ba3c4] focus-within:ring-2 focus-within:ring-indigo-500/10 dark:border-[#25283d] dark:focus-within:border-[#4b5278]'
                    }`}
                >
                    <textarea
                        ref={textareaRef}
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                if (query.trim() && !isLoading && !isLimitReached) {
                                    e.currentTarget.form?.requestSubmit();
                                }
                            }
                        }}
                        placeholder={
                            isLimitReached
                                ? 'Session limit reached. Reset session to begin anew.'
                                : 'Describe a genre, emotional mood, theme, or favorite book…'
                        }
                        disabled={isLoading || isLimitReached}
                        rows={1}
                        className="max-h-28 min-h-[38px] flex-1 resize-none bg-transparent px-2.5 py-1.5 font-matter text-sm leading-relaxed text-[#1e2033] outline-none placeholder:text-[#8f94a8] disabled:cursor-not-allowed dark:text-[#f3f4fa] dark:placeholder:text-[#6b728e]"
                        aria-label="Book recommendation prompt"
                    />

                    {/* Sarvam Signature Action Button */}
                    <button
                        type="submit"
                        disabled={isLoading || isLimitReached || !query.trim()}
                        className="btn-sarvam-filled grid h-9 w-9 shrink-0 place-items-center rounded-xl cursor-pointer disabled:cursor-not-allowed disabled:opacity-40"
                        aria-label="Send prompt"
                        title={isLimitReached ? 'Session limit reached' : 'Curate (Enter)'}
                    >
                        {isLoading ? (
                            <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                        ) : (
                            <SendIcon className="h-3.5 w-3.5" />
                        )}
                    </button>
                </div>

                {/* Footer Status Line */}
                <div className="mt-2.5 flex items-center justify-between px-2 font-matter text-[11px] text-[#8f94a8] dark:text-[#6b728e]">
                    <div className="flex items-center gap-1.5">
                        <span
                            className={`h-1.5 w-1.5 rounded-full ${
                                isLimitReached
                                    ? 'bg-amber-500'
                                    : remaining <= 2
                                    ? 'bg-amber-400'
                                    : 'bg-emerald-500'
                            }`}
                        />
                        <span>
                            {isLimitReached
                                ? 'Session limit reached (5 prompts used)'
                                : `${remaining} of ${queryLimit} prompts available`}
                        </span>
                        {isLimitReached && onReset && (
                            <button
                                type="button"
                                onClick={onReset}
                                className="ml-2 inline-flex items-center gap-1 font-medium text-indigo-700 hover:underline dark:text-indigo-400 cursor-pointer"
                            >
                                <RefreshIcon className="h-3 w-3" />
                                Reset Session
                            </button>
                        )}
                    </div>

                    <div className="hidden sm:flex items-center gap-1 opacity-75">
                        <span>Return ↵ to curate</span>
                    </div>
                </div>
            </form>
        </footer>
    );
}

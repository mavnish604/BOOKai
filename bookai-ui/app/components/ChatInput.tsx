import React from 'react';
import { SendIcon } from './Icons';

interface ChatInputProps {
    query: string;
    setQuery: (query: string) => void;
    handleSubmit: (e: React.FormEvent) => void;
    isLoading: boolean;
    queriesUsed: number;
    queryLimit: number;
}

export default function ChatInput({ query, setQuery, handleSubmit, isLoading, queriesUsed, queryLimit }: ChatInputProps) {
    const isLimitReached = queriesUsed >= queryLimit;

    return (
        <div className="relative flex-shrink-0 px-4 pb-4 pt-2 sm:px-6 sm:pb-5">
            <div className="pointer-events-none absolute inset-x-0 bottom-0 -z-10 h-28 bg-gradient-to-t from-[#f8f9ff] via-[#f8f9ff] to-transparent dark:from-[#090a12] dark:via-[#090a12]" />
            <form className="mx-auto max-w-3xl" onSubmit={handleSubmit}>
                <div className="flex items-end gap-2 rounded-[1.45rem] border border-slate-200 bg-white p-2 shadow-[0_10px_35px_rgba(59,70,130,0.12)] transition focus-within:border-indigo-300 focus-within:ring-4 focus-within:ring-indigo-100/70 dark:border-white/10 dark:bg-[#151622] dark:shadow-[0_10px_35px_rgba(0,0,0,0.28)] dark:focus-within:border-indigo-400/50 dark:focus-within:ring-indigo-500/10">
                    <textarea
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                e.currentTarget.form?.requestSubmit();
                            }
                        }}
                        placeholder={isLimitReached ? 'You’ve used all 5 prompts in this session.' : 'Ask about a book, genre, or reading mood…'}
                        disabled={isLoading || isLimitReached}
                        rows={1}
                        className="max-h-28 min-h-10 flex-1 resize-none bg-transparent py-2.5 text-sm leading-5 text-slate-800 outline-none placeholder:text-slate-400 disabled:cursor-not-allowed dark:text-slate-100 dark:placeholder:text-slate-500"
                        aria-label="Book recommendation prompt"
                    />
                    <button
                        type="submit"
                        disabled={isLoading || isLimitReached || !query.trim()}
                        className="mb-0.5 grid h-9 w-9 shrink-0 place-items-center rounded-full bg-gradient-to-br from-indigo-500 to-violet-600 text-white shadow-md shadow-indigo-500/25 transition hover:scale-105 hover:shadow-indigo-500/40 disabled:cursor-not-allowed disabled:scale-100 disabled:bg-slate-300 disabled:opacity-55 disabled:shadow-none dark:disabled:bg-slate-700"
                        aria-label="Send prompt"
                    >
                        {isLoading ? <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" /> : <SendIcon className="h-[17px] w-[17px]" />}
                    </button>
                </div>
                <p className="mt-2 text-center text-[11px] text-slate-400 dark:text-slate-500">
                    {isLimitReached ? 'Prompt limit reached — start a new browser session to continue.' : `${queryLimit - queriesUsed} of ${queryLimit} prompts left this session · Enter to send`}
                </p>
            </form>
        </div>
    );
}

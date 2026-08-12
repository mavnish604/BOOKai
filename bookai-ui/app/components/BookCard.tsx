"use client";

import React, { useEffect, useState } from 'react';
import { ArrowUpRightIcon, BookIcon, SparkleIcon } from './Icons';

interface BookCardProps {
    title: string;
    explanation: string;
    score: number;
}

export default function BookCard({ title, explanation, score }: BookCardProps) {
    const [summary, setSummary] = useState<string | null>(null);
    const [coverImage, setCoverImage] = useState<string | null>(null);
    const [isSummaryLoading, setIsSummaryLoading] = useState(false);
    const [isExpanded, setIsExpanded] = useState(false);
    const API_URL = process.env.NEXT_PUBLIC_BACKEND_URL;
    const percentage = Math.round(score * 100);

    useEffect(() => {
        let isMounted = true;

        async function fetchCover() {
            try {
                const response = await fetch(`/api/books/cover?title=${encodeURIComponent(title)}`);
                if (!response.ok) throw new Error(`Cover request failed with status ${response.status}`);
                const data = await response.json();
                if (isMounted && data.cover) setCoverImage(data.cover);
            } catch (error) {
                console.error('Error fetching book cover:', error);
            }
        }

        fetchCover();
        return () => { isMounted = false; };
    }, [title]);

    const fetchSummary = async () => {
        if (summary) {
            setIsExpanded((expanded) => !expanded);
            return;
        }

        setIsSummaryLoading(true);
        setIsExpanded(true);
        try {
            if (!API_URL) throw new Error('NEXT_PUBLIC_BACKEND_URL is not configured');
            const response = await fetch(`${API_URL}/summary`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ book_title: title }),
            });
            if (!response.ok) throw new Error(`Summary request failed with status ${response.status}`);
            const data = await response.json();
            setSummary(data.summary);
        } catch (error) {
            console.error('Error fetching summary:', error);
            setSummary('I couldn’t prepare a summary for this title right now.');
        } finally {
            setIsSummaryLoading(false);
        }
    };

    return (
        <article className="group overflow-hidden rounded-2xl border border-slate-200/80 bg-white p-3 shadow-sm transition duration-300 hover:-translate-y-0.5 hover:border-indigo-200 hover:shadow-lg hover:shadow-indigo-500/10 dark:border-white/8 dark:bg-[#151622] dark:hover:border-indigo-400/25 dark:hover:shadow-black/20">
            <div className="flex gap-3">
                <div className="relative h-[116px] w-[78px] shrink-0 overflow-hidden rounded-lg bg-gradient-to-br from-indigo-100 via-violet-100 to-fuchsia-100 shadow-sm dark:from-indigo-500/20 dark:via-violet-500/15 dark:to-fuchsia-500/15">
                    {coverImage ? (
                        <img src={coverImage} alt={`Cover of ${title}`} className="h-full w-full object-cover" referrerPolicy="no-referrer" onError={() => setCoverImage(null)} />
                    ) : (
                        <div className="grid h-full place-items-center text-indigo-400 dark:text-indigo-300"><BookIcon className="h-7 w-7" /></div>
                    )}
                    <div className="absolute inset-x-0 bottom-0 h-8 bg-gradient-to-t from-black/30 to-transparent" />
                </div>
                <div className="min-w-0 flex-1">
                    <div className="flex items-start justify-between gap-2">
                        <h3 className="line-clamp-2 text-sm font-semibold leading-5 text-slate-900 dark:text-white">{title}</h3>
                        <span className="shrink-0 rounded-full bg-indigo-50 px-2 py-1 text-[10px] font-semibold text-indigo-700 dark:bg-indigo-500/15 dark:text-indigo-300">{percentage}%</span>
                    </div>
                    <p className="mt-2 line-clamp-3 text-xs leading-5 text-slate-500 dark:text-slate-400">{explanation}</p>
                </div>
            </div>
            <div className="mt-3 flex items-center justify-between border-t border-slate-100 pt-2.5 dark:border-white/7">
                <button onClick={fetchSummary} disabled={isSummaryLoading} className="inline-flex items-center gap-1.5 rounded-lg px-1 text-xs font-medium text-indigo-600 transition hover:text-indigo-800 disabled:opacity-50 dark:text-indigo-300 dark:hover:text-indigo-200">
                    {isSummaryLoading ? <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent" /> : <SparkleIcon className="h-3.5 w-3.5" />}
                    {isSummaryLoading ? 'Thinking…' : summary && isExpanded ? 'Hide insight' : summary ? 'Show insight' : 'Get insight'}
                </button>
                <ArrowUpRightIcon className="h-4 w-4 text-slate-300 transition group-hover:text-indigo-400 dark:text-slate-600" />
            </div>
            <div className={`grid transition-[grid-template-rows,opacity,margin] duration-300 ${isExpanded ? 'mt-3 grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0'}`}>
                <div className="overflow-hidden">
                    <p className="rounded-xl bg-indigo-50/70 px-3 py-2.5 text-xs leading-5 text-slate-600 dark:bg-indigo-500/10 dark:text-slate-300">{summary}</p>
                </div>
            </div>
        </article>
    );
}

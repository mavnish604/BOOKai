"use client";

import React, { useEffect, useState } from 'react';
import {
    ArrowUpRightIcon,
    BookOpenIcon,
    CheckIcon,
    ChevronDownIcon,
    ChevronUpIcon,
    CopyIcon,
    SparklesIcon,
} from './Icons';

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
    const [isCopied, setIsCopied] = useState(false);

    const API_URL = process.env.NEXT_PUBLIC_BACKEND_URL;
    const percentage = Math.round(score * 100);

    // Fetch Google Books cover thumbnail
    useEffect(() => {
        let isMounted = true;

        async function fetchCover() {
            try {
                const response = await fetch(`/api/books/cover?title=${encodeURIComponent(title)}`);
                if (!response.ok) throw new Error(`Cover request failed with status ${response.status}`);
                const data = await response.json();
                if (isMounted && data.cover) {
                    setCoverImage(data.cover);
                }
            } catch (error) {
                console.error('Error fetching book cover:', error);
            }
        }

        fetchCover();
        return () => {
            isMounted = false;
        };
    }, [title]);

    // Fetch AI Summary / Deep Insight
    const fetchSummary = async () => {
        if (summary) {
            setIsExpanded((prev) => !prev);
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
            setSummary('Unable to prepare the deep summary for this book right now. Please try again.');
        } finally {
            setIsSummaryLoading(false);
        }
    };

    // Copy book title and brief to clipboard
    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(`"${title}" — ${explanation}`);
            setIsCopied(true);
            setTimeout(() => setIsCopied(false), 2000);
        } catch (err) {
            console.error('Failed to copy book details:', err);
        }
    };

    const goodreadsUrl = `https://www.goodreads.com/search?q=${encodeURIComponent(title)}`;
    const googleBooksUrl = `https://www.google.com/search?tbm=bks&q=${encodeURIComponent(title)}`;

    return (
        <article className="group relative flex flex-col justify-between overflow-hidden rounded-2xl border border-[#e2e5f1] bg-white p-4 shadow-[0_4px_16px_rgba(30,32,51,0.04)] transition-all duration-300 hover:-translate-y-0.5 hover:border-[#c4c9dd] hover:shadow-[0_8px_24px_rgba(30,32,51,0.08)] dark:border-[#25283d] dark:bg-[#12131e] dark:hover:border-[#383d5a]">
            {/* Card Main Body */}
            <div>
                <div className="flex gap-3.5">
                    {/* Realistic 3D Book Jacket */}
                    <div className="relative h-[130px] w-[88px] shrink-0 overflow-hidden rounded-lg bg-[#f0f2f9] shadow-[2px_6px_14px_rgba(30,32,51,0.12)] ring-1 ring-black/5 dark:bg-[#1a1b2b] dark:ring-white/10">
                        {coverImage ? (
                            <img
                                src={coverImage}
                                alt={`Cover for ${title}`}
                                className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                                referrerPolicy="no-referrer"
                                onError={() => setCoverImage(null)}
                            />
                        ) : (
                            /* Sarvam-Inspired Architectural Book Jacket */
                            <div className="flex h-full w-full flex-col justify-between bg-gradient-to-b from-[#2a2c3d] to-[#161724] p-2.5 text-white">
                                <div className="flex items-center justify-between opacity-60">
                                    <BookOpenIcon className="h-3 w-3 text-indigo-300" />
                                    <span className="font-matter text-[8px] font-medium uppercase tracking-widest text-[#a5b4fc]">
                                        Edition
                                    </span>
                                </div>
                                <div className="my-auto py-1 text-center">
                                    <p className="line-clamp-3 font-season-mix text-[11px] font-medium leading-tight text-[#f3f4fa]">
                                        {title}
                                    </p>
                                </div>
                                <div className="border-t border-white/10 pt-1 text-center">
                                    <span className="font-matter text-[7px] font-medium uppercase tracking-widest text-indigo-300">
                                        BookAI
                                    </span>
                                </div>
                            </div>
                        )}
                        {/* Book Spine Highlight Overlay */}
                        <div className="pointer-events-none absolute inset-y-0 left-0 w-1.5 bg-gradient-to-r from-black/30 via-white/15 to-transparent" />
                        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-6 bg-gradient-to-t from-black/25 to-transparent" />
                    </div>

                    {/* Meta & Synopsis */}
                    <div className="min-w-0 flex-1 flex flex-col justify-between">
                        <div>
                            <div className="flex items-start justify-between gap-2">
                                <h3
                                    className="line-clamp-2 font-season-mix text-[15px] font-medium leading-snug text-[#1e2033] dark:text-[#f3f4fa]"
                                    title={title}
                                >
                                    {title}
                                </h3>
                                {/* Match Badge */}
                                <span className="inline-flex shrink-0 items-center rounded-full border border-indigo-200/50 bg-[#eef2ff] px-2 py-0.5 font-matter text-[10px] font-semibold text-[#2c3360] dark:border-indigo-400/20 dark:bg-[#1a1e36] dark:text-[#c7d2fe]">
                                    {percentage > 0 ? `${percentage}% match` : 'Curated'}
                                </span>
                            </div>
                            <p className="mt-2 line-clamp-3 font-matter text-xs leading-relaxed text-[#555973] dark:text-[#9da3be]">
                                {explanation}
                            </p>
                        </div>

                        {/* Outbound Reference Badges */}
                        <div className="mt-3 flex items-center gap-2 font-matter">
                            <a
                                href={goodreadsUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-medium text-[#555973] transition hover:bg-[#f0f2f9] hover:text-[#1e2033] dark:text-[#9da3be] dark:hover:bg-[#1a1b2b] dark:hover:text-[#f3f4fa]"
                                title="Explore on Goodreads"
                            >
                                Goodreads
                                <ArrowUpRightIcon className="h-3 w-3" />
                            </a>
                            <span className="text-[#c7cbe0] dark:text-[#383d5a]">·</span>
                            <a
                                href={googleBooksUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-medium text-[#555973] transition hover:bg-[#f0f2f9] hover:text-[#1e2033] dark:text-[#9da3be] dark:hover:bg-[#1a1b2b] dark:hover:text-[#f3f4fa]"
                                title="Search Google Books"
                            >
                                Google Books
                                <ArrowUpRightIcon className="h-3 w-3" />
                            </a>
                        </div>
                    </div>
                </div>
            </div>

            {/* Bottom Actions Bar */}
            <div className="mt-3.5 flex items-center justify-between border-t border-[#e2e5f1] pt-2.5 dark:border-[#25283d]">
                <button
                    type="button"
                    onClick={fetchSummary}
                    disabled={isSummaryLoading}
                    className="inline-flex items-center gap-1.5 rounded-lg font-matter text-xs font-medium text-indigo-700 transition hover:text-indigo-900 disabled:opacity-50 dark:text-indigo-400 dark:hover:text-indigo-300 cursor-pointer"
                >
                    {isSummaryLoading ? (
                        <span className="h-3 w-3 animate-spin rounded-full border-2 border-indigo-600 border-t-transparent" />
                    ) : (
                        <SparklesIcon className="h-3.5 w-3.5" />
                    )}
                    <span>
                        {isSummaryLoading
                            ? 'Analyzing synopsis…'
                            : summary && isExpanded
                            ? 'Hide synopsis'
                            : summary
                            ? 'Read synopsis'
                            : 'AI Synopsis'}
                    </span>
                    {summary && (
                        isExpanded ? <ChevronUpIcon className="h-3 w-3" /> : <ChevronDownIcon className="h-3 w-3" />
                    )}
                </button>

                <button
                    type="button"
                    onClick={handleCopy}
                    className="inline-flex items-center gap-1 rounded-md p-1 text-[#8f94a8] transition hover:bg-[#f0f2f9] hover:text-[#1e2033] dark:text-[#6b728e] dark:hover:bg-[#1a1b2b] dark:hover:text-white cursor-pointer"
                    title={isCopied ? 'Copied' : 'Copy title & note'}
                    aria-label="Copy title"
                >
                    {isCopied ? (
                        <CheckIcon className="h-3.5 w-3.5 text-emerald-600" />
                    ) : (
                        <CopyIcon className="h-3.5 w-3.5" />
                    )}
                    {isCopied && <span className="text-[10px] text-emerald-600">Copied</span>}
                </button>
            </div>

            {/* Expandable Synopsis Drawer */}
            <div
                className={`grid transition-[grid-template-rows,opacity,margin] duration-300 ${
                    isExpanded ? 'mt-3 grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0'
                }`}
            >
                <div className="overflow-hidden">
                    <div className="rounded-xl border border-[#e2e5f1] bg-[#f9fafe] p-3.5 font-matter text-xs leading-relaxed text-[#1e2033] dark:border-[#25283d] dark:bg-[#0a0b12] dark:text-[#f3f4fa]">
                        <div className="mb-1.5 flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-indigo-700 dark:text-indigo-400">
                            <SparklesIcon className="h-3 w-3" />
                            BookAI Synopsis & Core Themes
                        </div>
                        <p>{summary}</p>
                    </div>
                </div>
            </div>
        </article>
    );
}

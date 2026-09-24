"use client";

import React, { useEffect, useSyncExternalStore } from 'react';
import { BookAiMark, MoonIcon, RefreshIcon, SparklesIcon, SunIcon } from './Icons';

const THEME_CHANGE_EVENT = 'bookai-theme-change';

function getThemeSnapshot() {
    const savedTheme = window.localStorage.getItem('bookai-theme');
    return savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches);
}

function subscribeToTheme(onStoreChange: () => void) {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    window.addEventListener(THEME_CHANGE_EVENT, onStoreChange);
    window.addEventListener('storage', onStoreChange);
    mediaQuery.addEventListener('change', onStoreChange);
    return () => {
        window.removeEventListener(THEME_CHANGE_EVENT, onStoreChange);
        window.removeEventListener('storage', onStoreChange);
        mediaQuery.removeEventListener('change', onStoreChange);
    };
}

interface HeaderProps {
    queriesUsed: number;
    queryLimit: number;
    onReset?: () => void;
}

export default function Header({ queriesUsed, queryLimit, onReset }: HeaderProps) {
    const isDark = useSyncExternalStore(subscribeToTheme, getThemeSnapshot, () => false);

    useEffect(() => {
        document.documentElement.classList.toggle('dark', isDark);
    }, [isDark]);

    const toggleTheme = () => {
        const nextTheme = !isDark;
        window.localStorage.setItem('bookai-theme', nextTheme ? 'dark' : 'light');
        window.dispatchEvent(new Event(THEME_CHANGE_EVENT));
    };

    const remaining = Math.max(queryLimit - queriesUsed, 0);
    const isLimitReached = queriesUsed >= queryLimit;

    return (
        <header className="sticky top-0 z-40 flex h-16 w-full flex-shrink-0 items-center justify-between border-b border-[#e2e5f1]/80 bg-[#f9fafe]/85 px-4 backdrop-blur-md transition-colors sm:px-8 dark:border-[#25283d] dark:bg-[#0a0b12]/85">
            {/* Left: Sarvam-style Wordmark & Mark */}
            <div className="flex items-center gap-3">
                <div className="relative flex h-8 w-8 items-center justify-center rounded-xl bg-white shadow-[inset_0_1px_1px_rgba(255,255,255,0.6),0_1px_3px_rgba(30,32,51,0.08)] ring-1 ring-[#1e2033]/10 dark:bg-[#1a1b2b] dark:ring-white/10">
                    <BookAiMark className="h-4.5 w-4.5 text-[#1e2033] dark:text-[#f3f4fa]" />
                </div>
                <div className="flex items-baseline gap-2">
                    <span className="font-matter text-[17px] font-semibold tracking-tight text-[#1e2033] dark:text-[#f3f4fa]">
                        BookAI
                    </span>
                    <span className="hidden rounded-full border border-indigo-200/50 bg-[#eef2ff] px-2 py-0.5 font-matter text-[10px] font-medium tracking-wide text-[#2e3776] sm:inline-block dark:border-indigo-400/20 dark:bg-[#1e2240] dark:text-[#c7d2fe]">
                        Sovereign
                    </span>
                </div>
            </div>

            {/* Middle: Subtle indicator (desktop) */}
            <div className="hidden md:flex items-center gap-6">
                <span className="font-matter text-xs font-medium tracking-wide text-[#555973] uppercase dark:text-[#9da3be]">
                    Platform for Curious Readers
                </span>
            </div>

            {/* Right: Actions in Sarvam Pill Style */}
            <div className="flex items-center gap-2.5">
                {/* Session Quota Pill */}
                <div
                    className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 font-matter text-xs font-medium transition-colors ${
                        isLimitReached
                            ? 'border-amber-300/80 bg-amber-50 text-amber-800 dark:border-amber-600/30 dark:bg-amber-950/20 dark:text-amber-300'
                            : 'border-[#e2e5f1] bg-white text-[#555973] shadow-xs dark:border-[#25283d] dark:bg-[#12131e] dark:text-[#9da3be]'
                    }`}
                >
                    <SparklesIcon className="h-3 w-3 text-indigo-500" />
                    <span className="hidden sm:inline">Prompts:</span>
                    <strong className="text-[#1e2033] dark:text-[#f3f4fa]">{queriesUsed}/{queryLimit}</strong>
                </div>

                {/* Reset Button (Sarvam Outline Style) */}
                {onReset && (
                    <button
                        type="button"
                        onClick={onReset}
                        className="btn-sarvam-outline hidden sm:inline-flex h-9 items-center gap-1.5 rounded-full px-4 font-matter text-xs font-medium cursor-pointer"
                        title="Start a fresh conversation"
                    >
                        <RefreshIcon className="h-3 w-3" />
                        <span>New Chat</span>
                    </button>
                )}

                {/* Theme Toggle */}
                <button
                    type="button"
                    onClick={toggleTheme}
                    className="grid h-9 w-9 place-items-center rounded-full border border-[#e2e5f1] bg-white text-[#555973] shadow-xs transition hover:border-[#c9cde0] hover:text-[#1e2033] focus:outline-none dark:border-[#25283d] dark:bg-[#12131e] dark:text-[#9da3be] dark:hover:border-[#383d5a] dark:hover:text-white cursor-pointer"
                    aria-label={`Switch to ${isDark ? 'light' : 'dark'} mode`}
                >
                    {isDark ? (
                        <SunIcon className="h-3.5 w-3.5 transition-transform hover:rotate-45" />
                    ) : (
                        <MoonIcon className="h-3.5 w-3.5 transition-transform hover:-rotate-12" />
                    )}
                </button>
            </div>
        </header>
    );
}

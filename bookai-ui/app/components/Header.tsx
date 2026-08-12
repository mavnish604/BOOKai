"use client";

import React, { useEffect, useSyncExternalStore } from 'react';
import Image from 'next/image';
import { MoonIcon, SparkleIcon, SunIcon } from './Icons';

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
}

export default function Header({ queriesUsed, queryLimit }: HeaderProps) {
    const isDark = useSyncExternalStore(subscribeToTheme, getThemeSnapshot, () => false);

    useEffect(() => {
        document.documentElement.classList.toggle('dark', isDark);
    }, [isDark]);

    const toggleTheme = () => {
        const nextTheme = !isDark;
        window.localStorage.setItem('bookai-theme', nextTheme ? 'dark' : 'light');
        window.dispatchEvent(new Event(THEME_CHANGE_EVENT));
    };

    return (
        <header className="relative z-10 flex h-[72px] flex-shrink-0 items-center justify-between border-b border-slate-200/70 bg-white/75 px-4 backdrop-blur-xl sm:px-7 dark:border-white/8 dark:bg-[#0d0e17]/75">
            <div className="flex items-center gap-3">
                <div className="h-14 w-[84px] shrink-0 overflow-hidden rounded-xl bg-[#090a12] shadow-md shadow-indigo-500/15 ring-1 ring-slate-200/80 dark:ring-white/10">
                    <Image
                        src="/bookai-logo.png"
                        alt="BookAI mascot logo"
                        width={1536}
                        height={1024}
                        priority
                        className="h-full w-full object-contain"
                    />
                </div>
                <div>
                    <h1 className="text-base font-semibold tracking-tight text-slate-900 dark:text-white">BookAI</h1>
                    <p className="hidden text-[11px] text-slate-500 sm:block dark:text-slate-400">Your reading companion</p>
                </div>
            </div>
            <div className="flex items-center gap-2 sm:gap-3">
                <div className="hidden items-center gap-1.5 rounded-full bg-indigo-50 px-3 py-1.5 text-xs font-medium text-indigo-700 sm:flex dark:bg-indigo-500/10 dark:text-indigo-300">
                    <SparkleIcon className="h-3.5 w-3.5" />
                    {queriesUsed}/{queryLimit} prompts
                </div>
                <button
                    type="button"
                    onClick={toggleTheme}
                    className="grid h-9 w-9 place-items-center rounded-full text-slate-600 transition hover:bg-slate-100 hover:text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 dark:text-slate-300 dark:hover:bg-white/10 dark:hover:text-white"
                    aria-label={`Switch to ${isDark ? 'light' : 'dark'} mode`}
                >
                    {isDark ? <SunIcon className="h-[18px] w-[18px]" /> : <MoonIcon className="h-[18px] w-[18px]" />}
                </button>
            </div>
        </header>
    );
}

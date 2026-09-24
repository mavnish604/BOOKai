"use client";

import React, { useState } from 'react';
import BookCard from './BookCard';
import { BookAiMark, CheckIcon, CopyIcon, UserIcon } from './Icons';

interface Recommendation {
    title: string;
    explanation: string;
    score: number;
}

interface MessageProps {
    sender: 'bot' | 'user';
    text: string;
    recommendations?: Recommendation[];
}

function formatInlineMarkdown(text: string, isBot: boolean) {
    return text.split(/(\*\*[^*]+\*\*)/g).map((part, index) => {
        if (part.startsWith('**') && part.endsWith('**')) {
            return (
                <strong
                    key={index}
                    className={`font-semibold ${
                        isBot ? 'text-[#1e2033] dark:text-[#f3f4fa]' : 'text-inherit font-medium'
                    }`}
                >
                    {part.slice(2, -2)}
                </strong>
            );
        }
        return part;
    });
}

function MessageText({ text, isBot }: { text: string; isBot: boolean }) {
    return (
        <div className="space-y-3 font-matter leading-relaxed">
            {text.split(/\n{2,}/).map((paragraph, index) => {
                if (paragraph.includes('\n- ') || paragraph.startsWith('- ')) {
                    const lines = paragraph.split('\n');
                    return (
                        <ul key={index} className="my-2 space-y-1.5 pl-1">
                            {lines.map((line, lIdx) => {
                                const cleanLine = line.replace(/^-\s*/, '');
                                return (
                                    <li key={lIdx} className="flex items-start gap-2.5">
                                        <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-indigo-600 dark:bg-indigo-400" />
                                        <span>{formatInlineMarkdown(cleanLine, isBot)}</span>
                                    </li>
                                );
                            })}
                        </ul>
                    );
                }
                return <p key={index}>{formatInlineMarkdown(paragraph, isBot)}</p>;
            })}
        </div>
    );
}

export default function ChatMessage({ sender, text, recommendations }: MessageProps) {
    const isBot = sender === 'bot';
    const [copied, setCopied] = useState(false);

    const handleCopyAll = async () => {
        try {
            await navigator.clipboard.writeText(text);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch (err) {
            console.error('Failed to copy message:', err);
        }
    };

    return (
        <article className={`group mb-8 flex w-full gap-3.5 animate-fade-in-up ${isBot ? 'items-start' : 'justify-end'}`}>
            {/* Sarvam-grade Bot Icon */}
            {isBot && (
                <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl border border-[#e2e5f1] bg-white text-indigo-700 shadow-xs dark:border-[#25283d] dark:bg-[#12131e] dark:text-indigo-400">
                    <BookAiMark className="h-4.5 w-4.5" />
                </div>
            )}

            {/* Message Body */}
            <div className={`min-w-0 ${isBot ? 'max-w-[calc(100%-3rem)] flex-1' : 'max-w-[85%] sm:max-w-[75%]'}`}>
                <div
                    className={`relative text-sm transition-colors ${
                        isBot
                            ? 'px-1 text-[#1e2033] dark:text-[#f3f4fa]'
                            : 'rounded-2xl rounded-tr-sm bg-[#1e2033] px-4 py-3 text-white shadow-[0_4px_16px_rgba(30,32,51,0.12)] dark:bg-[#1a1b2b] dark:text-[#f3f4fa]'
                    }`}
                >
                    <MessageText text={text} isBot={isBot} />

                    {/* Copy Utility Button */}
                    {isBot && (
                        <div className="mt-2.5 flex items-center gap-2 opacity-0 transition-opacity group-hover:opacity-100">
                            <button
                                type="button"
                                onClick={handleCopyAll}
                                className="inline-flex items-center gap-1 rounded-md px-2 py-0.5 font-matter text-[11px] text-[#8f94a8] hover:bg-[#f0f2f9] hover:text-[#1e2033] dark:hover:bg-[#1a1b2b] dark:hover:text-[#f3f4fa] cursor-pointer"
                                title="Copy response text"
                            >
                                {copied ? (
                                    <>
                                        <CheckIcon className="h-3 w-3 text-emerald-600" />
                                        <span className="text-emerald-600 font-medium">Copied</span>
                                    </>
                                ) : (
                                    <>
                                        <CopyIcon className="h-3 w-3" />
                                        <span>Copy text</span>
                                    </>
                                )}
                            </button>
                        </div>
                    )}
                </div>

                {/* Recommendations Grid */}
                {recommendations && recommendations.length > 0 && (
                    <div className="mt-4 grid gap-3.5 sm:grid-cols-2">
                        {recommendations.map((rec, index) => (
                            <BookCard key={`${rec.title}-${index}`} {...rec} />
                        ))}
                    </div>
                )}
            </div>

            {/* User Profile Avatar */}
            {!isBot && (
                <div className="mt-0.5 hidden h-8 w-8 shrink-0 items-center justify-center rounded-xl border border-[#e2e5f1] bg-[#f0f2f9] text-[#1e2033] sm:flex dark:border-[#25283d] dark:bg-[#1a1b2b] dark:text-[#f3f4fa]">
                    <UserIcon className="h-4 w-4" />
                </div>
            )}
        </article>
    );
}

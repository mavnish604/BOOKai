import React from 'react';
import BookCard from './BookCard';
import { BotIcon } from './Icons';

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
            return <strong key={index} className={isBot ? 'font-semibold text-slate-900 dark:text-white' : 'font-semibold text-white'}>{part.slice(2, -2)}</strong>;
        }
        return part;
    });
}

function MessageText({ text, isBot }: { text: string; isBot: boolean }) {
    return (
        <div className="space-y-3 whitespace-pre-wrap">
            {text.split(/\n{2,}/).map((paragraph, index) => (
                <p key={index}>{formatInlineMarkdown(paragraph, isBot)}</p>
            ))}
        </div>
    );
}

export default function ChatMessage({ sender, text, recommendations }: MessageProps) {
    const isBot = sender === 'bot';

    return (
        <article className={`mb-7 flex w-full gap-3 animate-fade-in-up ${isBot ? 'items-start' : 'justify-end'}`}>
            {isBot && (
                <div className="mt-1 grid h-8 w-8 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 text-white shadow-md shadow-indigo-500/20">
                    <BotIcon className="h-4 w-4" />
                </div>
            )}
            <div className={`min-w-0 ${isBot ? 'max-w-[calc(100%-2.75rem)] flex-1' : 'max-w-[85%] sm:max-w-[70%]'}`}>
                <div className={`text-sm leading-6 ${isBot ? 'px-1 py-1 text-slate-700 dark:text-slate-200' : 'rounded-2xl rounded-tr-sm bg-gradient-to-br from-indigo-500 to-violet-600 px-4 py-3 text-white shadow-md shadow-indigo-500/15'}`}>
                    <MessageText text={text} isBot={isBot} />
                </div>
                {recommendations && recommendations.length > 0 && (
                    <div className="mt-4 grid gap-3 sm:grid-cols-2">
                        {recommendations.map((rec, index) => (
                            <BookCard key={`${rec.title}-${index}`} {...rec} />
                        ))}
                    </div>
                )}
            </div>
        </article>
    );
}

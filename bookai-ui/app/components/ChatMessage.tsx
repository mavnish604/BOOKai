import React from 'react';
import BookCard from './BookCard';

interface Recommendation {
    title: string;
    explanation: string;
    score: number;
}

interface MessageProps {
    sender: string;
    text: string;
    recommendations?: Recommendation[];
}

export default function ChatMessage({ sender, text, recommendations }: MessageProps) {
    const isBot = sender === 'bot';

    return (
        <div className={`flex w-full ${isBot ? 'justify-start' : 'justify-end'} mb-6`}>
            <div className={`flex max-w-[90%] md:max-w-[80%] gap-3 ${isBot ? 'flex-row' : 'flex-row-reverse'}`}>

                {/* Avatar */}
                <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold shadow-sm ${isBot ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-600'
                    }`}>
                    {isBot ? 'B' : 'U'}
                </div>

                {/* Message Bubble & Content */}
                <div className="flex flex-col gap-2">

                    {/* Text Bubble */}
                    <div className={`px-4 py-3 rounded-2xl shadow-sm text-[15px] leading-relaxed ${isBot
                            ? 'bg-white text-gray-800 border-gray-100 border rounded-tl-none'
                            : 'bg-blue-600 text-white rounded-tr-none'
                        }`}>
                        {text}
                    </div>

                    {/* Recommendations Rendered OUTSIDE the bubble for better layout */}
                    {recommendations && recommendations.length > 0 && (
                        <div className="flex flex-col gap-3 mt-1 w-full animate-fade-in-up">
                            {recommendations.map((rec, index) => (
                                <BookCard
                                    key={index}
                                    title={rec.title}
                                    explanation={rec.explanation}
                                    score={rec.score}
                                />
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

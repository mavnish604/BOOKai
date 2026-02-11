import React from 'react';

export default function Header() {
    return (
        <header className="flex-shrink-0 bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between shadow-sm z-10">
            <div className="flex items-center gap-2">
                <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center text-white font-bold text-xl">
                    B
                </div>
                <h1 className="text-xl font-bold text-gray-800 tracking-tight">BOOKai</h1>
            </div>
            <div className="text-sm text-gray-500">
                AI-Powered Recommendations
            </div>
        </header>
    );
}

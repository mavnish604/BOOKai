import type { SVGProps } from 'react';

export type IconProps = SVGProps<SVGSVGElement>;

function Icon({ children, className = "h-5 w-5", ...props }: IconProps) {
    return (
        <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.75"
            strokeLinecap="round"
            strokeLinejoin="round"
            className={className}
            aria-hidden="true"
            {...props}
        >
            {children}
        </svg>
    );
}

/** BookAI Brand Vector Mark: Open book meeting a luminous intelligence spark */
export function BookAiMark({ className = "h-6 w-6", ...props }: SVGProps<SVGSVGElement>) {
    return (
        <svg
            viewBox="0 0 32 32"
            fill="none"
            className={className}
            aria-hidden="true"
            {...props}
        >
            <defs>
                <linearGradient id="bookai-grad" x1="2" y1="4" x2="30" y2="28" gradientUnits="userSpaceOnUse">
                    <stop stopColor="#6366f1" />
                    <stop offset="0.5" stopColor="#8b5cf6" />
                    <stop offset="1" stopColor="#a855f7" />
                </linearGradient>
                <linearGradient id="bookai-glow" x1="16" y1="6" x2="16" y2="26" gradientUnits="userSpaceOnUse">
                    <stop stopColor="#c084fc" stopOpacity="0.8" />
                    <stop offset="1" stopColor="#6366f1" stopOpacity="0.2" />
                </linearGradient>
            </defs>
            {/* Left Page */}
            <path
                d="M5 8.5C5 7.12 6.12 6 7.5 6H14.5C15.33 6 16 6.67 16 7.5V24C16 24 14.5 23 11 23H7.5C6.12 23 5 24.12 5 25.5V8.5Z"
                fill="url(#bookai-grad)"
                fillOpacity="0.15"
                stroke="currentColor"
                strokeWidth="1.75"
                strokeLinecap="round"
                strokeLinejoin="round"
            />
            {/* Right Page */}
            <path
                d="M27 8.5C27 7.12 25.88 6 24.5 6H17.5C16.67 6 16 6.67 16 7.5V24C16 24 17.5 23 21 23H24.5C25.88 23 27 24.12 27 25.5V8.5Z"
                fill="url(#bookai-grad)"
                fillOpacity="0.25"
                stroke="currentColor"
                strokeWidth="1.75"
                strokeLinecap="round"
                strokeLinejoin="round"
            />
            {/* Spine Fold */}
            <path
                d="M16 7.5V24.5"
                stroke="currentColor"
                strokeWidth="1.75"
                strokeLinecap="round"
            />
            {/* Luminous Core Sparkle */}
            <path
                d="M16 10L17.2 13.8L21 15L17.2 16.2L16 20L14.8 16.2L11 15L14.8 13.8L16 10Z"
                fill="url(#bookai-grad)"
            />
        </svg>
    );
}

export function SparkleIcon(props: IconProps) {
    return (
        <Icon {...props}>
            <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" opacity="0.35" />
            <path d="m12 5 1.8 4.2L18 11l-4.2 1.8L12 17l-1.8-4.2L6 11l4.2-1.8L12 5Z" />
        </Icon>
    );
}

export function SparklesIcon(props: IconProps) {
    return (
        <Icon {...props}>
            <path d="m12 3-1.5 5.5L5 10l5.5 1.5L12 17l1.5-5.5L19 10l-5.5-1.5L12 3Z" />
            <path d="m19 16-.8 2.2L16 19l2.2.8.8 2.2.8-2.2L22 19l-2.2-.8-.8-2.2Z" />
        </Icon>
    );
}

export function BookIcon(props: IconProps) {
    return (
        <Icon {...props}>
            <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z" />
            <path d="M6 6h10" />
            <path d="M6 10h7" />
        </Icon>
    );
}

export function BookOpenIcon(props: IconProps) {
    return (
        <Icon {...props}>
            <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
            <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
        </Icon>
    );
}

export function BotIcon(props: IconProps) {
    return (
        <Icon {...props}>
            <rect width="16" height="14" x="4" y="6" rx="3.5" />
            <path d="M12 2v4" />
            <circle cx="9" cy="13" r="1" fill="currentColor" />
            <circle cx="15" cy="13" r="1" fill="currentColor" />
            <path d="M10 16.5h4" strokeWidth="1.5" />
        </Icon>
    );
}

export function UserIcon(props: IconProps) {
    return (
        <Icon {...props}>
            <circle cx="12" cy="8" r="4" />
            <path d="M20 21a8 8 0 1 0-16 0" />
        </Icon>
    );
}

export function MoonIcon(props: IconProps) {
    return (
        <Icon {...props}>
            <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" />
        </Icon>
    );
}

export function SunIcon(props: IconProps) {
    return (
        <Icon {...props}>
            <circle cx="12" cy="12" r="4" />
            <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
        </Icon>
    );
}

export function SendIcon(props: IconProps) {
    return (
        <Icon {...props}>
            <path d="M3.7 11.3 20.3 3.7c.9-.4 1.8.5 1.4 1.4l-7.6 16.6c-.4.9-1.7.9-2 0l-2.4-5.8-5.8-2.4c-.9-.4-.9-1.7-.2-2.2Z" />
            <path d="m11.7 13.9 4.3-4.3" />
        </Icon>
    );
}

export function ArrowUpRightIcon(props: IconProps) {
    return (
        <Icon {...props}>
            <path d="M7 17 17 7M8 7h9v9" />
        </Icon>
    );
}

export function ChevronDownIcon(props: IconProps) {
    return (
        <Icon {...props}>
            <path d="m6 9 6 6 6-6" />
        </Icon>
    );
}

export function ChevronUpIcon(props: IconProps) {
    return (
        <Icon {...props}>
            <path d="m18 15-6-6-6 6" />
        </Icon>
    );
}

export function CopyIcon(props: IconProps) {
    return (
        <Icon {...props}>
            <rect width="13" height="13" x="8" y="8" rx="2" ry="2" />
            <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
        </Icon>
    );
}

export function CheckIcon(props: IconProps) {
    return (
        <Icon {...props}>
            <path d="M20 6 9 17l-5-5" />
        </Icon>
    );
}

export function RefreshIcon(props: IconProps) {
    return (
        <Icon {...props}>
            <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
            <path d="M21 3v5h-5" />
            <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
            <path d="M3 21v-5h5" />
        </Icon>
    );
}

export function SearchIcon(props: IconProps) {
    return (
        <Icon {...props}>
            <circle cx="11" cy="11" r="8" />
            <path d="m21 21-4.3-4.3" />
        </Icon>
    );
}

export function CompassIcon(props: IconProps) {
    return (
        <Icon {...props}>
            <circle cx="12" cy="12" r="10" />
            <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" />
        </Icon>
    );
}

export function BookmarkIcon(props: IconProps) {
    return (
        <Icon {...props}>
            <path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16z" />
        </Icon>
    );
}

/** Sarvam AI-Inspired Sacred Geometry Motif */
export function SarvamMotif({ className = "h-8 w-auto", ...props }: SVGProps<SVGSVGElement>) {
    return (
        <svg
            viewBox="0 0 120 32"
            fill="none"
            className={className}
            aria-hidden="true"
            {...props}
        >
            <path
                d="M60 2L64 8H76L70 14L74 24L60 18L46 24L50 14L44 8H56L60 2Z"
                stroke="currentColor"
                strokeWidth="1.25"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="text-indigo-600 dark:text-indigo-400"
            />
            {/* Left Gateway Wing */}
            <path
                d="M44 14C35 14 26 12 16 6M44 18C33 18 22 17 8 12M44 22C30 22 18 22 2 20"
                stroke="currentColor"
                strokeWidth="1"
                strokeLinecap="round"
                strokeOpacity="0.4"
            />
            {/* Right Gateway Wing */}
            <path
                d="M76 14C85 14 94 12 104 6M76 18C87 18 98 17 112 12M76 22C90 22 102 22 118 20"
                stroke="currentColor"
                strokeWidth="1"
                strokeLinecap="round"
                strokeOpacity="0.4"
            />
            <circle cx="60" cy="13" r="2.5" fill="currentColor" className="text-indigo-500" />
            <circle cx="48" cy="14" r="1.2" fill="currentColor" opacity="0.6" />
            <circle cx="72" cy="14" r="1.2" fill="currentColor" opacity="0.6" />
            <circle cx="28" cy="13" r="1" fill="currentColor" opacity="0.3" />
            <circle cx="92" cy="13" r="1" fill="currentColor" opacity="0.3" />
        </svg>
    );
}


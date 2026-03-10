import { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import { BarChart3, Bot, Settings, LifeBuoy, ChevronsLeft, ChevronsRight, ScanSearch, XCircle } from 'lucide-react';

const Sidebar = ({ isSidebarOpen, setIsSidebarOpen }) => {
    const navItems = [
        { icon: <BarChart3 size={20} />, label: 'Smart Trades', path: '/' },
        { icon: <ScanSearch size={20} />, label: 'Scanner', path: '/scanner' },
        { icon: <Bot size={20} />, label: 'Active Positions', path: '/positions' },
        { icon: <Settings size={20} />, label: 'Settings', path: '/settings' },
    ];

    return (
        <div className={`transition-all duration-200 ease-in-out bg-slate-950 border-r border-slate-800 flex flex-col ${isSidebarOpen ? 'w-64' : 'w-20'}`}>
            {/* Header */}
            <div className="p-4 border-b border-slate-800 flex items-center" style={{ minHeight: '88px' }}>
                {isSidebarOpen && (
                    <div className="flex-1">
                        <h1 className="text-xl font-black tracking-tighter text-white">Trading OS</h1>
                        <p className="text-xs text-slate-500">v1.0 Pro</p>
                    </div>
                )}
                <button 
                    onClick={() => setIsSidebarOpen(!isSidebarOpen)} 
                    className="text-slate-500 hover:text-white p-2 rounded-md hover:bg-slate-800"
                >
                    {isSidebarOpen ? <ChevronsLeft size={20} /> : <ChevronsRight size={20} />}
                </button>
            </div>
            
            {/* Navigation */}
            <nav className="p-4 flex-grow">
                <ul>
                    {navItems.map((item, index) => (
                        <li key={index} className="mb-2">
                            <NavLink
                                to={item.path}
                                className={({ isActive }) => 
                                    `flex items-center gap-4 px-4 py-3 rounded-lg transition-colors text-sm font-bold ${
                                        isActive
                                            ? 'bg-blue-600/10 text-blue-300 border border-blue-500/20'
                                            : 'text-slate-500 hover:bg-slate-800/30 hover:text-slate-300'
                                    } ${!isSidebarOpen && 'justify-center'}`
                                }
                            >
                                {item.icon}
                                {isSidebarOpen && <span className="truncate">{item.label}</span>}
                            </NavLink>
                        </li>
                    ))}
                </ul>
            </nav>

            {/* Footer */}
            <div className="p-4 border-t border-slate-800">
                <ul>
                    <li>
                        <NavLink
                            to="/help"
                            className={({ isActive }) => 
                                `flex items-center gap-4 px-4 py-3 rounded-lg transition-colors text-sm font-bold ${
                                    isActive
                                        ? 'bg-blue-600/10 text-blue-300 border border-blue-500/20'
                                        : 'text-slate-500 hover:bg-slate-800/30 hover:text-slate-300'
                                } ${!isSidebarOpen && 'justify-center'}`
                            }
                        >
                            <LifeBuoy size={20} />
                            {isSidebarOpen && <span className="truncate">Help & Support</span>}
                        </NavLink>
                    </li>
                </ul>
            </div>
        </div>
    );
};

export default Sidebar;
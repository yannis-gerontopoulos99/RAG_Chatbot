'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, FileText, Loader, AlertCircle } from 'lucide-react';

interface Message {
  id: number;
  type: 'user' | 'assistant' | 'system' | 'error';
  content: string;
  sources?: Source[];
  confidence?: number;
  timestamp: string;
}

interface Source {
  title: string;
  source: string;
  relevance_score: number;
  snippet: string;
}

interface ChatQuery {
  question: string;
  conversation_history?: Array<{[key: string]: string}>;
  session_id?: string;
}

const ChatInterface = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const sessionId = useRef(`session_${Date.now()}`);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Test backend connection on component mount
  useEffect(() => {
    const testConnection = async () => {
      try {
        const response = await fetch('http://localhost:8000/health');
        if (response.ok) {
          setConnectionError(null);
        } else {
          setConnectionError('Backend is running but not responding correctly');
        }
      } catch (error) {
        setConnectionError('Cannot connect to backend. Make sure it\'s running on port 8000.');
      }
    };

    testConnection();
    // Test connection every 30 seconds
    const interval = setInterval(testConnection, 30000);
    return () => clearInterval(interval);
  }, []);

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    // Get conversation history (last 6 messages for context)
    const conversationHistory = messages.slice(-6).map(msg => ({
      [msg.type]: msg.content
    }));

    const queryData: ChatQuery = {
      question: inputMessage,
      conversation_history: conversationHistory,
      session_id: sessionId.current
    };

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(queryData)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      setMessages(prev => [...prev, {
        id: Date.now(),
        type: 'assistant',
        content: result.answer,
        sources: result.sources || [],
        confidence: result.confidence_score,
        timestamp: result.timestamp || new Date().toISOString()
      }]);

      setConnectionError(null);
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [...prev, {
        id: Date.now(),
        type: 'error',
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error occurred'}`,
        timestamp: new Date().toISOString()
      }]);
      setConnectionError('Failed to send message. Check your backend connection.');
    } finally {
      setIsLoading(false);
    }

    setInputMessage('');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <div className="bg-white/80 backdrop-blur-sm border-b border-slate-200 p-4 shadow-sm">
        <div className="flex items-center justify-between max-w-4xl mx-auto">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <Bot className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-slate-800">RAG Chatbot</h1>
              <div className="flex items-center space-x-2 text-sm">
                <div className={`w-2 h-2 rounded-full ${!connectionError ? 'bg-green-400' : 'bg-red-400'}`}></div>
                <span className="text-slate-600">
                  {!connectionError ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            </div>
          </div>
        </div>
        
        {connectionError && (
          <div className="max-w-4xl mx-auto mt-3">
            <div className="bg-red-50 border border-red-200 rounded-lg p-3 flex items-center space-x-2">
              <AlertCircle className="w-5 h-5 text-red-500" />
              <span className="text-red-700 text-sm">{connectionError}</span>
            </div>
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <Bot className="w-16 h-16 text-slate-300 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-slate-600 mb-2">
                Welcome to RAG Chatbot
              </h3>
              <p className="text-slate-500">
                Ask me anything about the documents in the knowledge base!
              </p>
            </div>
          )}
          
          {messages.map((message) => (
            <div key={message.id} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`flex space-x-3 max-w-3xl ${message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                  message.type === 'user' 
                    ? 'bg-blue-500' 
                    : message.type === 'error'
                    ? 'bg-red-500'
                    : message.type === 'system'
                    ? 'bg-yellow-500'
                    : 'bg-slate-600'
                }`}>
                  {message.type === 'user' ? (
                    <User className="w-4 h-4 text-white" />
                  ) : message.type === 'system' ? (
                    <Loader className="w-4 h-4 text-white animate-spin" />
                  ) : (
                    <Bot className="w-4 h-4 text-white" />
                  )}
                </div>
                
                <div className={`rounded-2xl px-4 py-3 ${
                  message.type === 'user'
                    ? 'bg-blue-500 text-white'
                    : message.type === 'error'
                    ? 'bg-red-50 text-red-800 border border-red-200'
                    : message.type === 'system'
                    ? 'bg-yellow-50 text-yellow-800 border border-yellow-200'
                    : 'bg-white border border-slate-200 shadow-sm'
                }`}>
                  <div className="prose prose-sm max-w-none">
                    <p className="whitespace-pre-wrap mb-0">{message.content}</p>
                  </div>
                  
                  {message.confidence && (
                    <div className="mt-2 text-xs text-slate-500">
                      Confidence: {Math.round(message.confidence * 100)}%
                    </div>
                  )}
                  
                  {message.sources && message.sources.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-slate-200">
                      <div className="text-xs font-medium text-slate-600 mb-2 flex items-center">
                        <FileText className="w-3 h-3 mr-1" />
                        Sources ({message.sources.length})
                      </div>
                      <div className="space-y-2">
                        {message.sources.map((source, idx) => (
                          <div key={idx} className="bg-slate-50 rounded-lg p-2 border border-slate-200">
                            <div className="flex items-start justify-between mb-1">
                              <div className="font-medium text-xs text-slate-700">
                                {source.title}
                              </div>
                              <div className="text-xs text-slate-500">
                                {Math.round(source.relevance_score * 100)}%
                              </div>
                            </div>
                            <div className="text-xs text-slate-600 line-clamp-2">
                              {source.snippet}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  <div className={`text-xs mt-2 ${
                    message.type === 'user' ? 'text-blue-200' : 'text-slate-400'
                  }`}>
                    {formatTimestamp(message.timestamp)}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="bg-white/80 backdrop-blur-sm border-t border-slate-200 p-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex space-x-3">
            <div className="flex-1">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask me anything about the documents..."
                className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                rows={1}
                disabled={isLoading}
              />
            </div>
            <button
              onClick={sendMessage}
              disabled={!inputMessage.trim() || isLoading}
              className="bg-blue-500 hover:bg-blue-600 disabled:bg-slate-300 disabled:cursor-not-allowed text-white rounded-lg px-4 py-3 transition-colors duration-200 flex items-center justify-center"
            >
              {isLoading ? (
                <Loader className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
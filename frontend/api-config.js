// ===================================================================
// API Configuration — Secure Setup
// ===================================================================
// ⚠️ WARNING: NEVER put API keys directly in frontend JavaScript!
//
// This file provides a secure pattern for connecting to AI APIs.
// API keys must be stored server-side or in environment variables.
//
// WRONG ❌ (Never do this):
//   const API_KEY = "sk-abc123...";
//   const API_KEY = "AIzaSy...";
//
// RIGHT ✅ (Use a backend proxy):
//   Send requests to YOUR backend → backend calls AI API with key
// ===================================================================

const APIConfig = (() => {
  'use strict';

  // --- Configuration ---
  // Set your backend proxy URL here (NOT the AI provider's URL directly)
  const BACKEND_PROXY_URL = '/api/chat'; // Your server endpoint

  // Allowed origins for API communication
  const ALLOWED_ORIGINS = [
    window.location.origin,
    // Add your production domain here, e.g.:
    // 'https://yourdomain.com',
  ];

  // Request timeout in milliseconds
  const REQUEST_TIMEOUT_MS = 30000;

  // --- Security Checks ---

  /**
   * Detects if any API key patterns are accidentally exposed in the page.
   * Call this on page load as a safety net.
   */
  function auditForExposedKeys() {
    const dangerousPatterns = [
      /sk-[a-zA-Z0-9]{20,}/,           // OpenAI keys
      /AIza[a-zA-Z0-9_-]{35}/,         // Google API keys
      /sk-ant-[a-zA-Z0-9_-]{40,}/,     // Anthropic keys
      /ghp_[a-zA-Z0-9]{36}/,           // GitHub tokens
      /Bearer\s+[a-zA-Z0-9._~+\/=-]{20,}/, // Bearer tokens
      /[a-f0-9]{32,64}/,               // Generic hex secrets (loose)
    ];

    // Scan all script tags
    const scripts = document.querySelectorAll('script:not([src])');
    scripts.forEach((script, index) => {
      dangerousPatterns.forEach(pattern => {
        if (pattern.test(script.textContent)) {
          console.error(
            `🚨 [SECURITY AUDIT] Possible API key detected in inline <script> #${index}! ` +
            `Pattern: ${pattern.toString()}. Remove it immediately!`
          );
        }
      });
    });

    // Scan meta tags
    const metas = document.querySelectorAll('meta');
    metas.forEach(meta => {
      const content = meta.getAttribute('content') || '';
      dangerousPatterns.slice(0, 4).forEach(pattern => {
        if (pattern.test(content)) {
          console.error(
            `🚨 [SECURITY AUDIT] Possible API key detected in <meta> tag! ` +
            `Name: "${meta.getAttribute('name')}". Remove it immediately!`
          );
        }
      });
    });

    // Scan localStorage
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        const value = localStorage.getItem(key) || '';
        dangerousPatterns.slice(0, 4).forEach(pattern => {
          if (pattern.test(value)) {
            console.error(
              `🚨 [SECURITY AUDIT] Possible API key detected in localStorage["${key}"]! ` +
              `Remove it and rotate the key immediately!`
            );
          }
        });
      }
    } catch (e) {
      // localStorage might be blocked
    }

    console.info('✅ [SECURITY AUDIT] API key exposure scan complete.');
  }

  /**
   * Sends a chat message to the backend proxy securely.
   * The backend proxy should forward the request to the AI provider
   * with the API key stored server-side.
   *
   * @param {string} message - The user message
   * @param {Array} history - Conversation history array
   * @returns {Promise<string>} - The bot response text
   */
  async function sendToBackend(message, history = []) {
    // Validate origin
    if (!ALLOWED_ORIGINS.includes(window.location.origin)) {
      throw new Error('Unauthorized origin');
    }

    // Create abort controller for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
      const response = await fetch(BACKEND_PROXY_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // CSRF protection — backend should validate this
          'X-Requested-With': 'XMLHttpRequest',
        },
        credentials: 'same-origin', // Send cookies for session auth
        signal: controller.signal,
        body: JSON.stringify({
          message: message,
          history: history,
          // No API key here! The backend adds it.
          timestamp: Date.now(),
        }),
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        if (response.status === 429) {
          throw new Error('RATE_LIMITED');
        }
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();

      // Validate response structure
      if (!data || typeof data.reply !== 'string') {
        throw new Error('Invalid response format from server');
      }

      return data.reply;

    } catch (error) {
      clearTimeout(timeoutId);

      if (error.name === 'AbortError') {
        throw new Error('Request timed out. Please try again.');
      }
      throw error;
    }
  }

  /**
   * Example: How to integrate with app.js
   *
   * In app.js, replace the predefined response logic:
   *
   * BEFORE (current — predefined responses):
   *   const botText = getBotResponse(cleanText);
   *
   * AFTER (API-connected):
   *   try {
   *     const botText = await APIConfig.sendToBackend(cleanText, messages);
   *   } catch (error) {
   *     if (error.message === 'RATE_LIMITED') {
   *       showToast('تم تجاوز الحد المسموح. استنى شوية.', 'error');
   *     } else {
   *       showToast('حصل مشكلة في الاتصال. حاول تاني.', 'error');
   *     }
   *   }
   */

  // --- Public API ---
  return {
    sendToBackend,
    auditForExposedKeys,
    // Expose config for testing only in dev
    ...(window.location.hostname === 'localhost' ? {
      _config: { BACKEND_PROXY_URL, REQUEST_TIMEOUT_MS },
    } : {}),
  };
})();

// Run security audit on page load
document.addEventListener('DOMContentLoaded', () => {
  APIConfig.auditForExposedKeys();
});

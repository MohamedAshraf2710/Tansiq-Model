// ===================================================================
// المرشد الأكاديمي — Academic Advisor Chatbot Frontend
// ===================================================================

document.addEventListener('DOMContentLoaded', () => {
  'use strict';

  // Initialize Lucide icons
  if (window.lucide && typeof lucide.createIcons === 'function') {
    lucide.createIcons();
  }

  // ===== Security: DOM Clobbering Protection =====
  // Safely get DOM elements — prevents DOM clobbering attacks where
  // attacker-controlled HTML elements override getElementById results
  function safeGetElement(id) {
    const el = document.getElementById(id);
    // Verify it's actually a DOM Element, not a clobbered property
    if (el && el instanceof HTMLElement) return el;
    return null;
  }

  // ===== DOM References (with clobbering protection) =====
  const chatMessages = safeGetElement('chatMessages');
  const chatInput = safeGetElement('chatInput');
  const sendBtn = safeGetElement('sendBtn');
  const welcomeScreen = safeGetElement('welcomeScreen');
  const featureCards = safeGetElement('featureCards');
  const sidebar = safeGetElement('sidebar');
  const sidebarToggle = safeGetElement('sidebarToggle');
  const toggleSidebarBtn = safeGetElement('toggleSidebarBtn');
  const sidebarOverlay = safeGetElement('sidebarOverlay');
  const newChatBtn = safeGetElement('newChatBtn');
  const chatHistory = safeGetElement('chatHistory');
  const particlesContainer = safeGetElement('particles');

  const profileModal = safeGetElement('profileModal');
  const profileForm = safeGetElement('profileForm');
  const editProfileBtn = safeGetElement('editProfileBtn');

  // ===== State =====
  let messages = [];
  let isTyping = false;
  let chatSessions = loadSessionsFromStorage();
  let currentSessionId = null;

  // ===== Security: HTML Sanitizer =====
  function sanitizeHTML(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  // ===== Security: Validate & load sessions from localStorage =====
  function loadSessionsFromStorage() {
    try {
      const raw = localStorage.getItem('chatSessions');
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      // Validate each session structure
      return parsed.filter(s =>
        s && typeof s.id === 'string' &&
        typeof s.title === 'string' &&
        Array.isArray(s.messages)
      ).map(s => ({
        ...s,
        // Sanitize title on load
        title: s.title.substring(0, 50),
        messages: s.messages.filter(m =>
          m && (m.role === 'user' || m.role === 'bot') &&
          typeof m.text === 'string' &&
          typeof m.time === 'string'
        ).map(m => ({
          ...m,
          // Cap message text length
          text: m.text.substring(0, 5000)
        }))
      }));
    } catch (e) {
      console.warn('Invalid chat sessions in localStorage, resetting.');
      localStorage.removeItem('chatSessions');
      return [];
    }
  }

  // ===== Security: Input validation =====
  const MAX_INPUT_LENGTH = 2000;
  const MAX_SESSIONS = 50;          // Limit stored sessions
  const MAX_MESSAGES_PER_SESSION = 200; // Limit messages per session

  // ===== Security: Rate Limiting =====
  const RATE_LIMIT_CONFIG = {
    minIntervalMs: 1500,        // Minimum 1.5s between messages
    windowMs: 60000,            // Sliding window: 1 minute
    maxPerWindow: 20,           // Max 20 messages per minute
    burstLimit: 5,              // Max 5 messages in burst window
    burstWindowMs: 5000,        // Burst window: 5 seconds
    cooldownMs: 10000,          // Cooldown penalty: 10 seconds
  };

  const rateLimitState = {
    timestamps: [],             // Array of message timestamps
    lastMessageTime: 0,         // Last message timestamp
    cooldownUntil: 0,           // Cooldown expiry timestamp
    warningCount: 0,            // Consecutive warning count
  };

  /**
   * Check if the user is currently rate limited.
   * Returns { allowed: boolean, reason: string, waitMs: number }
   */
  function checkRateLimit() {
    const now = Date.now();

    // Check if in cooldown penalty
    if (now < rateLimitState.cooldownUntil) {
      const waitMs = rateLimitState.cooldownUntil - now;
      return {
        allowed: false,
        reason: `لقد تجاوزت الحد المسموح. انتظر ${Math.ceil(waitMs / 1000)} ثانية.`,
        waitMs
      };
    }

    // Check minimum interval between messages
    const timeSinceLast = now - rateLimitState.lastMessageTime;
    if (timeSinceLast < RATE_LIMIT_CONFIG.minIntervalMs) {
      const waitMs = RATE_LIMIT_CONFIG.minIntervalMs - timeSinceLast;
      return {
        allowed: false,
        reason: 'استنى لحظة قبل ما تبعت رسالة تانية...',
        waitMs
      };
    }

    // Clean old timestamps outside the sliding window
    rateLimitState.timestamps = rateLimitState.timestamps.filter(
      t => now - t < RATE_LIMIT_CONFIG.windowMs
    );

    // Check sliding window limit (messages per minute)
    if (rateLimitState.timestamps.length >= RATE_LIMIT_CONFIG.maxPerWindow) {
      rateLimitState.warningCount++;
      // Apply cooldown penalty after repeated violations
      if (rateLimitState.warningCount >= 3) {
        rateLimitState.cooldownUntil = now + RATE_LIMIT_CONFIG.cooldownMs;
        rateLimitState.warningCount = 0;
      }
      return {
        allowed: false,
        reason: `وصلت للحد الأقصى (${RATE_LIMIT_CONFIG.maxPerWindow} رسالة في الدقيقة). استنى شوية.`,
        waitMs: RATE_LIMIT_CONFIG.cooldownMs
      };
    }

    // Check burst detection (too many messages too fast)
    const recentTimestamps = rateLimitState.timestamps.filter(
      t => now - t < RATE_LIMIT_CONFIG.burstWindowMs
    );
    if (recentTimestamps.length >= RATE_LIMIT_CONFIG.burstLimit) {
      rateLimitState.warningCount++;
      if (rateLimitState.warningCount >= 2) {
        rateLimitState.cooldownUntil = now + RATE_LIMIT_CONFIG.cooldownMs;
        rateLimitState.warningCount = 0;
      }
      return {
        allowed: false,
        reason: 'بتبعت رسائل بسرعة كبيرة! استنى كام ثانية. ⏳',
        waitMs: RATE_LIMIT_CONFIG.burstWindowMs
      };
    }

    return { allowed: true, reason: '', waitMs: 0 };
  }

  /** Record a sent message timestamp */
  function recordMessageSent() {
    const now = Date.now();
    rateLimitState.timestamps.push(now);
    rateLimitState.lastMessageTime = now;
    // Reset warning count on successful send
    rateLimitState.warningCount = Math.max(0, rateLimitState.warningCount - 1);
  }

  // ===== Toast Notification System =====
  function showToast(message, type = 'warning', durationMs = 4000) {
    // Remove existing toast
    const existing = document.getElementById('rateLimitToast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.id = 'rateLimitToast';
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.style.cssText = `
      position: fixed;
      bottom: 100px;
      left: 50%;
      transform: translateX(-50%) translateY(20px);
      background: ${type === 'warning'
        ? 'linear-gradient(135deg, #b85c00, #d4a017)'
        : 'linear-gradient(135deg, #c0392b, #e74c3c)'};
      color: #fff;
      padding: 12px 24px;
      border-radius: 12px;
      font-size: 0.9rem;
      font-weight: 600;
      font-family: var(--font-primary);
      direction: rtl;
      z-index: 9999;
      box-shadow: 0 8px 32px rgba(0,0,0,0.4);
      opacity: 0;
      transition: opacity 0.3s ease, transform 0.3s ease;
      display: flex;
      align-items: center;
      gap: 8px;
      max-width: 90vw;
    `;
    toast.textContent = message;

    document.body.appendChild(toast);

    // Animate in
    requestAnimationFrame(() => {
      toast.style.opacity = '1';
      toast.style.transform = 'translateX(-50%) translateY(0)';
    });

    // Auto dismiss
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(-50%) translateY(20px)';
      setTimeout(() => toast.remove(), 300);
    }, durationMs);
  }

  /** Show visual cooldown on the send button */
  function applySendCooldown(waitMs) {
    sendBtn.disabled = true;
    chatInput.disabled = true;
    chatInput.style.opacity = '0.5';

    const originalPlaceholder = chatInput.placeholder;
    const seconds = Math.ceil(waitMs / 1000);
    chatInput.placeholder = `⏳ انتظر ${seconds} ثانية...`;

    // Countdown update
    let remaining = seconds;
    const countdownInterval = setInterval(() => {
      remaining--;
      if (remaining > 0) {
        chatInput.placeholder = `⏳ انتظر ${remaining} ثانية...`;
      }
    }, 1000);

    setTimeout(() => {
      clearInterval(countdownInterval);
      chatInput.disabled = false;
      chatInput.style.opacity = '1';
      chatInput.placeholder = originalPlaceholder;
      sendBtn.disabled = chatInput.value.trim() === '';
      chatInput.focus();
    }, waitMs);
  }

  // ===== Security Utilities =====
  function sanitizeUrl(url) {
    try {
      const parsed = new URL(url);
      if (!['http:', 'https:'].includes(parsed.protocol)) return '';
      return parsed.href;
    } catch { return ''; }
  }

  // ===== Profile Management =====
  function loadProfile() {
    try {
      return JSON.parse(localStorage.getItem('studentProfile'));
    } catch (e) {
      return null;
    }
  }

  function showProfileModal() {
    const profile = loadProfile();
    if (profile) {
      if (safeGetElement('studentScore')) safeGetElement('studentScore').value = profile.score || '';
      if (safeGetElement('studentTrack')) safeGetElement('studentTrack').value = profile.track || '';
      if (safeGetElement('studentGov')) safeGetElement('studentGov').value = profile.gov || '';
      if (safeGetElement('studentGender')) safeGetElement('studentGender').value = profile.gender || '';
      if (safeGetElement('studentPriority')) safeGetElement('studentPriority').value = profile.priority || 'غير محدد';
    }
    if (profileModal) profileModal.classList.add('active');
  }

  function hideProfileModal() {
    if (profileModal) profileModal.classList.remove('active');
  }

  if (profileForm) {
    profileForm.addEventListener('submit', (e) => {
      e.preventDefault();
      
      let govVal = safeGetElement('studentGov').value.replace(/[<>"'&]/g, '').trim().substring(0, 50);
      let scoreVal = safeGetElement('studentScore').value;
      const scoreNum = parseFloat(scoreVal);
      
      if (isNaN(scoreNum) || scoreNum < 50 || scoreNum > 420) {
        showToast('المجموع لازم يكون بين 50 و 420', 'error');
        return;
      }
      
      const profile = {
        score: scoreVal,
        track: safeGetElement('studentTrack').value,
        gov: govVal,
        gender: safeGetElement('studentGender').value,
        priority: safeGetElement('studentPriority').value
      };
      
      if (!profile.track || !profile.gov || !profile.gender) {
        showToast('من فضلك املأ كل البيانات المطلوبة', 'error');
        return;
      }

      localStorage.setItem('studentProfile', JSON.stringify(profile));
      hideProfileModal();
      showToast('تم حفظ البيانات بنجاح!', 'success', 3000);
    });
  }

  if (editProfileBtn) {
    editProfileBtn.addEventListener('click', showProfileModal);
  }

  // Show modal if no profile exists
  if (!loadProfile()) {
    setTimeout(showProfileModal, 1000);
  }


  // ===== Time Formatter =====
  function formatTime(date) {
    return new Intl.DateTimeFormat('ar-EG', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: true,
    }).format(date);
  }

  // ===== Create Particles =====
  function createParticles() {
    const count = 20;
    for (let i = 0; i < count; i++) {
      const particle = document.createElement('div');
      particle.classList.add('particle');
      const size = Math.random() * 4 + 2;
      particle.style.width = `${size}px`;
      particle.style.height = `${size}px`;
      particle.style.left = `${Math.random() * 100}%`;
      particle.style.animationDuration = `${Math.random() * 15 + 10}s`;
      particle.style.animationDelay = `${Math.random() * 10}s`;
      particlesContainer.appendChild(particle);
    }
  }
  createParticles();

  // ===== Auto-resize textarea =====
  chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
    sendBtn.disabled = chatInput.value.trim() === '';
  });

  // ===== Send Message =====
  async function sendMessage(text) {
    if (!text || !text.trim()) return;

    // Security: Rate Limit Check
    const rateCheck = checkRateLimit();
    if (!rateCheck.allowed) {
      showToast(rateCheck.reason, rateCheck.waitMs > 5000 ? 'error' : 'warning');
      applySendCooldown(Math.min(rateCheck.waitMs, RATE_LIMIT_CONFIG.cooldownMs));
      return;
    }

    // Security: enforce input length limit
    let cleanText = text.trim();
    if (cleanText.length > MAX_INPUT_LENGTH) {
      cleanText = cleanText.substring(0, MAX_INPUT_LENGTH);
    }

    // Hide welcome screen
    if (welcomeScreen) {
      welcomeScreen.style.display = 'none';
    }

    const userMsg = {
      role: 'user',
      text: cleanText,
      time: new Date(),
    };
    messages.push(userMsg);
    appendMessage(userMsg);

    // Record for rate limiting
    recordMessageSent();

    // Clear input
    chatInput.value = '';
    chatInput.style.height = 'auto';
    sendBtn.disabled = true;

    // Show typing indicator
    showTyping();

    // Setup Backend Payload
    if (!currentSessionId) {
      currentSessionId = (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : Date.now().toString() + Math.random().toString(36).substring(2);
    }
    
    const profile = loadProfile() || {};
    
    const payload = {
      session_id: currentSessionId,
      question: cleanText,
      student_score: profile.score ? parseFloat(profile.score) : null,
      student_gender: profile.gender || null,
      student_gov: profile.gov || null,
      track: profile.track || null,
      priority: profile.priority || "غير محدد",
      interests: [] // Can be updated if we add interests tracking
    };

    try {
      const res = await fetch(window.APIConfig?.BACKEND_PROXY_URL || "http://localhost:8000/predict", {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "X-Requested-With": "XMLHttpRequest"
        },
        credentials: "same-origin",
        body: JSON.stringify(payload)
      });
      
      if (!res.ok) throw new Error("API Network error");
      
      const data = await res.json();
      hideTyping();
      
      let botText = data.answer || "عذراً، حدث خطأ أثناء معالجة الطلب.";
      
      // Check for 75 wishes table
      if (data.wishes_75 && data.wishes_75.length > 0) {
        botText += "\n\n### 📋 قائمة الكليات المقترحة ليك:\n";
        botText += "| م | الكلية | المحافظة | الحد الأدنى |\n";
        botText += "|---|-------|----------|-------------|\n";
        const displayLimit = Math.min(15, data.wishes_75.length);
        for (let i = 0; i < displayLimit; i++) {
          const w = data.wishes_75[i];
          const safeUrl = w.url ? sanitizeUrl(w.url) : '';
          const safeFaculty = (w.faculty || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
          const facultyName = safeUrl ? `<a href="${safeUrl}" target="_blank" rel="noopener noreferrer">${safeFaculty}</a>` : safeFaculty;
          botText += `| ${i+1} | ${facultyName} | ${w.governorate} | ${w.min_score} |\n`;
        }
        if (data.wishes_75.length > displayLimit) {
          botText += `\n*... تم إخفاء باقي ${data.wishes_75.length - displayLimit} رغبة لتسهيل القراءة.*`;
        }
      }

      const botMsg = {
        role: 'bot',
        text: botText,
        time: new Date(),
      };
      messages.push(botMsg);
      appendMessage(botMsg);
      saveCurrentSession();
      
    } catch (err) {
      console.error("Backend Error:", err);
      hideTyping();
      const botMsg = {
        role: 'bot',
        text: "عذراً، الخادم لا يستجيب حالياً. تأكد من تشغيل الـ GP_ml Backend.",
        time: new Date(),
      };
      messages.push(botMsg);
      appendMessage(botMsg);
    }
  }

  // ===== Append Message to DOM =====
  function appendMessage(msg) {
    const div = document.createElement('div');
    div.classList.add('message', msg.role);

    const avatar = document.createElement(msg.role === 'bot' ? 'img' : 'div');
    avatar.classList.add('message-avatar');
    if (msg.role === 'bot') {
      avatar.src = 'advisor_avatar.png';
      avatar.alt = 'المرشد';
    } else {
      avatar.textContent = '👤';
    }

    const content = document.createElement('div');
    content.classList.add('message-content');

    const bubble = document.createElement('div');
    bubble.classList.add('message-bubble');

    // SECURITY FIX: User messages use textContent (no HTML parsing)
    // Bot messages use parseMarkdown (safe — content is from predefined responses only)
    if (msg.role === 'user') {
      bubble.textContent = msg.text;
    } else {
      bubble.innerHTML = parseMarkdown(msg.text);
    }

    const time = document.createElement('div');
    time.classList.add('message-time');
    time.textContent = formatTime(msg.time);

    content.appendChild(bubble);
    content.appendChild(time);

    div.appendChild(avatar);
    div.appendChild(content);

    chatMessages.appendChild(div);
    scrollToBottom();
  }

  // ===== Simple Markdown Parser =====
  function parseMarkdown(text) {
    let html = text
      // Escape HTML
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      // Headers
      .replace(/^### (.+)$/gm, '<h4 style="color:var(--accent-300);margin:8px 0 4px;font-size:0.95rem">$1</h4>')
      .replace(/^## (.+)$/gm, '<h3 style="color:var(--accent-200);margin:12px 0 6px;font-size:1.1rem">$1</h3>')
      // Bold
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      // Italic
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      // Tables
      .replace(/^\|(.+)\|$/gm, (match) => {
        const cells = match.split('|').filter(c => c.trim());
        if (cells.every(c => /^[-\s:]+$/.test(c.trim()))) return ''; // separator row
        const tag = 'td';
        const row = cells.map(c => `<${tag} style="padding:4px 10px;border:1px solid var(--border-subtle)">${c.trim()}</${tag}>`).join('');
        return `<tr>${row}</tr>`;
      })
      // Unordered list
      .replace(/^- (.+)$/gm, '<li style="margin:2px 0;padding-right:4px">$1</li>')
      // Ordered list
      .replace(/^\d+\. (.+)$/gm, '<li style="margin:2px 0;padding-right:4px;list-style-type:decimal">$1</li>')
      // Line breaks
      .replace(/\n\n/g, '<br/><br/>')
      .replace(/\n/g, '<br/>');

    // Wrap <tr> in <table>
    if (html.includes('<tr>')) {
      html = html.replace(
        /(<tr>[\s\S]*?<\/tr>(?:<br\/>)?)+/g,
        '<table style="border-collapse:collapse;margin:8px 0;width:100%;font-size:0.9rem">$&</table>'
      );
    }

    // Wrap <li> in <ul>
    html = html.replace(
      /(<li[\s\S]*?<\/li>(?:<br\/>)?)+/g,
      '<ul style="padding-right:20px;margin:4px 0">$&</ul>'
    );

    return html;
  }

  // ===== Typing Indicator =====
  function showTyping() {
    if (isTyping) return;
    isTyping = true;

    const div = document.createElement('div');
    div.classList.add('typing-indicator');
    div.id = 'typingIndicator';

    const avatar = document.createElement('img');
    avatar.src = 'advisor_avatar.png';
    avatar.alt = '';
    avatar.classList.add('message-avatar');
    avatar.style.border = '2px solid var(--accent-500)';

    const dots = document.createElement('div');
    dots.classList.add('typing-dots');
    for (let i = 0; i < 3; i++) {
      const dot = document.createElement('div');
      dot.classList.add('typing-dot');
      dots.appendChild(dot);
    }

    div.appendChild(avatar);
    div.appendChild(dots);
    chatMessages.appendChild(div);
    scrollToBottom();
  }

  function hideTyping() {
    isTyping = false;
    const indicator = document.getElementById('typingIndicator');
    if (indicator) indicator.remove();
  }

  // ===== Scroll =====
  function scrollToBottom() {
    requestAnimationFrame(() => {
      chatMessages.scrollTop = chatMessages.scrollHeight;
    });
  }

  // ===== Session Management =====
  function saveCurrentSession() {
    if (!currentSessionId || messages.length === 0) return;

    const firstUserMsg = messages.find(m => m.role === 'user');
    const title = firstUserMsg ? firstUserMsg.text.substring(0, 40) : 'محادثة جديدة';

    const session = {
      id: currentSessionId,
      title: title + (firstUserMsg && firstUserMsg.text.length > 40 ? '...' : ''),
      messages: messages.map(m => ({ ...m, time: m.time.toISOString() })),
      updatedAt: new Date().toISOString(),
    };

    const idx = chatSessions.findIndex(s => s.id === currentSessionId);
    if (idx >= 0) {
      chatSessions[idx] = session;
    } else {
      chatSessions.unshift(session);
    }

    // Security: Limit total sessions to prevent localStorage overflow
    if (chatSessions.length > MAX_SESSIONS) {
      chatSessions = chatSessions.slice(0, MAX_SESSIONS);
    }

    // Security: Limit messages per session
    if (session.messages.length > MAX_MESSAGES_PER_SESSION) {
      session.messages = session.messages.slice(-MAX_MESSAGES_PER_SESSION);
    }

    // Security: Safe localStorage write with quota protection
    try {
      localStorage.setItem('chatSessions', JSON.stringify(chatSessions));
    } catch (e) {
      if (e.name === 'QuotaExceededError' || e.code === 22) {
        // Storage full — remove oldest sessions and retry
        console.warn('localStorage quota exceeded, removing old sessions.');
        chatSessions = chatSessions.slice(0, Math.ceil(chatSessions.length / 2));
        try {
          localStorage.setItem('chatSessions', JSON.stringify(chatSessions));
        } catch (e2) {
          console.error('Failed to save sessions even after cleanup.');
          localStorage.removeItem('chatSessions');
        }
      }
    }
    renderChatHistory();
  }

  function loadSession(sessionId) {
    const session = chatSessions.find(s => s.id === sessionId);
    if (!session) return;

    currentSessionId = session.id;
    messages = session.messages.map(m => ({ ...m, time: new Date(m.time) }));

    // Clear messages area
    chatMessages.innerHTML = '';
    if (welcomeScreen) {
      chatMessages.appendChild(welcomeScreen);
      welcomeScreen.style.display = 'none';
    }

    messages.forEach(msg => appendMessage(msg));
    renderChatHistory();
  }

  function renderChatHistory() {
    chatHistory.innerHTML = '';
    chatSessions.forEach(session => {
      const item = document.createElement('div');
      item.classList.add('history-item');
      if (session.id === currentSessionId) item.classList.add('active');

      // SECURITY FIX: Use DOM API instead of innerHTML to prevent XSS
      const iconSpan = document.createElement('span');
      iconSpan.classList.add('history-icon');
      iconSpan.textContent = '💬';

      const titleSpan = document.createElement('span');
      titleSpan.textContent = session.title; // textContent = safe, no HTML parsing

      item.appendChild(iconSpan);
      item.appendChild(titleSpan);
      item.addEventListener('click', () => loadSession(session.id));
      chatHistory.appendChild(item);
    });
  }

  function startNewChat() {
    currentSessionId = null;
    messages = [];
    chatMessages.innerHTML = '';
    if (welcomeScreen) {
      chatMessages.appendChild(welcomeScreen);
      welcomeScreen.style.display = '';
    }
    renderChatHistory();
    chatInput.focus();
  }

  // ===== Event Listeners =====

  // Send button
  sendBtn.addEventListener('click', () => {
    sendMessage(chatInput.value);
  });

  // Enter to send (Shift+Enter for new line)
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (chatInput.value.trim()) {
        sendMessage(chatInput.value);
      }
    }
  });

  // Feature cards
  featureCards.addEventListener('click', (e) => {
    const card = e.target.closest('.feature-card');
    if (card) {
      const msg = card.dataset.message;
      if (msg) sendMessage(msg);
    }
  });

  // Quick topic buttons
  document.querySelectorAll('.quick-topic-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const msg = btn.dataset.message;
      if (msg) {
        sendMessage(msg);
        // Close sidebar on mobile
        if (window.innerWidth <= 1024) {
          sidebar.classList.add('collapsed');
          sidebarOverlay.classList.remove('active');
        }
      }
    });
  });

  // Sidebar toggle (mobile)
  if (sidebarToggle) {
    sidebarToggle.addEventListener('click', () => {
      sidebar.classList.toggle('collapsed');
      sidebarOverlay.classList.toggle('active');
    });
  }

  // Toggle sidebar from header
  if (toggleSidebarBtn) {
    toggleSidebarBtn.addEventListener('click', () => {
      sidebar.classList.toggle('collapsed');
      if (window.innerWidth <= 1024) {
        sidebarOverlay.classList.toggle('active');
      }
    });
  }

  // Sidebar overlay click
  if (sidebarOverlay) {
    sidebarOverlay.addEventListener('click', () => {
      sidebar.classList.add('collapsed');
      sidebarOverlay.classList.remove('active');
    });
  }

  // New chat
  newChatBtn.addEventListener('click', startNewChat);

  // ===== Initial Setup =====
  renderChatHistory();
  chatInput.focus();

  // Responsive: collapse sidebar on mobile by default
  if (window.innerWidth <= 1024) {
    sidebar.classList.add('collapsed');
  }
});

// ==========================================================================
// Festival Iguassu Inova 2026 — Chatbot Frontend
// ==========================================================================

(function () {
  const form = document.getElementById('chat-form');
  const input = document.getElementById('user-input');
  const btnSend = document.getElementById('btn-send');
  const messages = document.getElementById('messages');
  const typingIndicator = document.getElementById('typing-indicator');
  const suggestBtns = document.querySelectorAll('.suggest-btn');
  const openChatBtn = document.getElementById('open-chat');
  const mobilePanel = document.getElementById('mobile-panel');
  const menuToggle = document.querySelector('.menu-toggle');

  // ---- Utilitários ---------------------------------------------------------
  function scrollToBottom() {
    messages.scrollTop = messages.scrollHeight;
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function addMessage(text, role) {
    const wrapper = document.createElement('div');
    wrapper.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = `message-avatar ${role}-avatar`;
    avatar.textContent = role === 'bot' ? '▲' : 'U';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    // Converte quebras de linha em parágrafos
    const html = text.split('\n\n').map(p => `<p>${escapeHtml(p)}</p>`).join('');
    bubble.innerHTML = html;

    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);
    messages.appendChild(wrapper);

    scrollToBottom();
    return wrapper;
  }

  function setTyping(show) {
    typingIndicator.style.display = show ? 'flex' : 'none';
    if (show) scrollToBottom();
  }

  function setInputEnabled(enabled) {
    input.disabled = !enabled;
    btnSend.disabled = !enabled;
    if (enabled) input.focus();
  }

  // ---- Eventos -------------------------------------------------------------
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;

    addMessage(text, 'user');
    input.value = '';
    setInputEnabled(false);
    setTyping(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });
      if (!res.ok) throw new Error('Erro no servidor');
      const data = await res.json();
      setTyping(false);
      addMessage(data.answer || 'Desculpe, não consegui processar sua pergunta.', 'bot');
    } catch (err) {
      setTyping(false);
      addMessage('Desculpe, ocorreu um erro ao conectar com o servidor. Tente novamente.', 'bot');
    } finally {
      setInputEnabled(true);
    }
  });

  // Perguntas sugeridas
  suggestBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const q = btn.getAttribute('data-q');
      if (!q) return;
      input.value = q;
      form.dispatchEvent(new Event('submit'));
    });
  });

  // Abrir chat direto do hero
  if (openChatBtn) {
    openChatBtn.addEventListener('click', (e) => {
      e.preventDefault();
      document.getElementById('chat').scrollIntoView({ behavior: 'smooth' });
      input.focus();
    });
  }

  // Menu mobile
  if (menuToggle && mobilePanel) {
    let open = false;
    menuToggle.addEventListener('click', () => {
      open = !open;
      mobilePanel.classList.toggle('open', open);
      menuToggle.setAttribute('aria-expanded', open);
    });
  }

  // Inicialização
  input.focus();
})();

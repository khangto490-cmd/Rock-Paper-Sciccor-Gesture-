/**
 * ==========================================================================
 * OẲN TÙ TÌ AI - LOGIC ĐIỀU KHIỂN CHUYỂN MÀN HÌNH & ÂM THANH
 * 
 * Yêu cầu: Pure JavaScript, Không thư viện ngoài, Hoạt động 100% Offline.
 * Giữ nguyên các tên hàm và biến cốt lõi: startGame, onEnterGameScreen,
 * onEnterHomeScreen, GameApp, playSynthSound.
 * ==========================================================================
 */

// Đóng gói logic trong namespace GameApp để tránh xung đột toàn cục
const GameApp = (() => {
  // 1. Quản lý trạng thái giao diện (UI State)
  const state = {
    currentScreen: 'screen-home',
    isAudioEnabled: true
  };

  // 2. Tham chiếu các phần tử DOM
  const dom = {
    screens: document.querySelectorAll('.screen'),
    screenHome: document.getElementById('screen-home'),
    screenGame: document.getElementById('screen-game'),
    btnPlay: document.getElementById('btn-play'),
    btnBack: document.getElementById('btn-back'),
    toast: document.getElementById('toast'),
    handCards: document.querySelectorAll('.hand-card')
  };

  // 3. Hệ thống âm thanh Synthesizer tổng hợp (Web Audio API - Offline 100%)
  let audioCtx = null;

  function getAudioContext() {
    if (!audioCtx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (AudioContextClass) {
        audioCtx = new AudioContextClass();
      }
    }
    if (audioCtx && audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    return audioCtx;
  }

  function playSynthSound(type = 'click') {
    if (!state.isAudioEnabled) return;
    try {
      const ctx = getAudioContext();
      if (!ctx) return;

      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);

      const now = ctx.currentTime;

      if (type === 'start') {
        // Âm thanh khi bấm CHƠI hoặc chuyển CONFIRM (Rising chime)
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(320, now);
        osc.frequency.exponentialRampToValueAtTime(880, now + 0.22);
        gain.gain.setValueAtTime(0.25, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.3);
        osc.start(now);
        osc.stop(now + 0.3);
      } else if (type === 'click') {
        // Tiếng tick click ngắn gọn
        osc.type = 'sine';
        osc.frequency.setValueAtTime(550, now);
        osc.frequency.exponentialRampToValueAtTime(220, now + 0.08);
        gain.gain.setValueAtTime(0.2, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.08);
        osc.start(now);
        osc.stop(now + 0.08);
      } else if (type === 'hover') {
        // Tiếng rê chuột cực nhẹ
        osc.type = 'sine';
        osc.frequency.setValueAtTime(420, now);
        gain.gain.setValueAtTime(0.04, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.04);
        osc.start(now);
        osc.stop(now + 0.04);
      }
    } catch (e) {
      // Bỏ qua nếu trình duyệt yêu cầu tương tác người dùng trước khi phát âm thanh
    }
  }

  // 4. Quản lý chuyển đổi màn hình (Screen Navigation)
  function switchScreen(targetScreenId) {
    dom.screens.forEach(screen => {
      if (screen.id === targetScreenId) {
        screen.classList.add('active');
      } else {
        screen.classList.remove('active');
      }
    });
    state.currentScreen = targetScreenId;

    // Kích hoạt lifecycle callback
    if (targetScreenId === 'screen-game') {
      onEnterGameScreen();
    } else if (targetScreenId === 'screen-home') {
      onEnterHomeScreen();
    }
  }

  // 5. Hiển thị thông báo nhanh (Toast)
  let toastTimer = null;
  function showToast(message, duration = 2500) {
    if (!dom.toast) return;
    dom.toast.textContent = message;
    dom.toast.classList.add('show');

    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      dom.toast.classList.remove('show');
    }, duration);
  }

  // 6. Xử lý logic khi bấm nút CHƠI ở Trang chủ
  function handlePlayButtonClick() {
    playSynthSound('start');
    switchScreen('screen-game');
  }

  // Hàm alias theo đúng tên biến quy định
  function startGame() {
    handlePlayButtonClick();
  }

  // 7. Vòng đời màn hình (Lifecycle hooks)
  function onEnterGameScreen() {
    console.log('[GameApp] Vào màn hình chơi.');
    if (typeof GameUI !== 'undefined') {
      GameUI.start();
    }
  }

  function onEnterHomeScreen() {
    console.log('[GameApp] Quay về trang chủ.');
    if (typeof GameUI !== 'undefined') {
      GameUI.stop();
    }
  }

  /**
   * Khởi động webcam dự phòng (nếu dùng trực tiếp trên client)
   */
  async function initWebcam(videoElement) {
    // Được xử lý qua luồng /video của server.py
  }

  /**
   * Phán đoán kết quả (dự phòng, JS không tự tính khi chơi thật)
   */
  function determineWinner(player, ai) {
    if (player === ai) return 'draw';
    if (
      (player === 'rock' && ai === 'scissors') ||
      (player === 'scissors' && ai === 'paper') ||
      (player === 'paper' && ai === 'rock')
    ) {
      return 'player';
    }
    return 'ai';
  }

  // 8. Đăng ký các sự kiện tương tác
  function initEvents() {
    // Nút CHƠI ở trang chủ
    if (dom.btnPlay) {
      dom.btnPlay.addEventListener('click', handlePlayButtonClick);
      dom.btnPlay.addEventListener('mouseenter', () => playSynthSound('hover'));
    }

    // Nút Quay lại trang chủ
    if (dom.btnBack) {
      dom.btnBack.addEventListener('click', () => {
        playSynthSound('click');
        switchScreen('screen-home');
      });
    }

    // Tương tác nhẹ với 3 bàn tay trên trang chủ
    dom.handCards.forEach(card => {
      card.addEventListener('mouseenter', () => playSynthSound('hover'));
      card.addEventListener('click', () => {
        playSynthSound('click');
        const name = card.querySelector('.hand-name')?.textContent || '';
        showToast(`Chiêu thức: ${name}`);
      });
    });
  }

  // 9. Hàm khởi động
  function init() {
    console.log('[GameApp] Khởi tạo hệ thống giao diện thành công.');
    initEvents();

    // Tự động vào màn chơi nếu URL có ?demo=1
    if (typeof DemoState !== 'undefined' && DemoState.isDemoMode()) {
      switchScreen('screen-game');
    }
  }

  return {
    init,
    switchScreen,
    showToast,
    determineWinner,
    playSynthSound,
    startGame,
    onEnterGameScreen,
    onEnterHomeScreen
  };
})();

// Xuất các hàm cấp toàn cục theo quy định
function startGame() {
  GameApp.startGame();
}

function onEnterGameScreen() {
  GameApp.onEnterGameScreen();
}

function onEnterHomeScreen() {
  GameApp.onEnterHomeScreen();
}

function playSynthSound(type) {
  GameApp.playSynthSound(type);
}

// Khởi chạy khi DOM đã sẵn sàng
document.addEventListener('DOMContentLoaded', () => {
  GameApp.init();
});

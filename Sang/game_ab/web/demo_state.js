/**
 * ==========================================================================
 * OẲN TÙ TÌ AI - BỘ PHÁT TRẠNG THÁI GIẢ LẬP (DEMO MODE)
 * 
 * Kích hoạt khi URL có tham số: ?demo=1
 * Mô phỏng chính xác hợp đồng dữ liệu của server.py:
 * - Tọa độ: a = {x: 0.2, y: 0.5}, b = {x: 0.8, y: 0.5}
 * - box_r = 0.13, band = 0.09, gate_f = 0.25
 * - Vòng lặp vô hạn luân phiên các kịch bản:
 *   Ván 1: WAIT 2s -> NEED_A 2s (a_hold 0->1) -> GO 1s (passed=false) -> GO 1s (passed=true) -> off_line 0.5s -> CONFIRM 0.2s -> RESULT 3s.
 *   Ván 2: NEED_A 1s -> GO warn=true 0.5s -> NEED_A 1.5s -> GO -> CONFIRM -> CANCEL 3s.
 *   Các ván tiếp theo xoay vòng: ai thắng, hòa, không xác định.
 * ==========================================================================
 */

const DemoState = (() => {
  // Kiểm tra tham số ?demo=1 trên URL
  const isDemo = new URLSearchParams(window.location.search).get('demo') === '1';

  // Hằng số hình học theo luật A -> B
  const CONFIG = {
    a: { x: 0.2, y: 0.5 },
    b: { x: 0.8, y: 0.5 },
    box_r: 0.13,
    band: 0.09,
    gate_f: 0.25,
    fps: 30
  };

  // Điểm số và số ván hủy giả lập
  const score = {
    player: 0,
    ai: 0,
    draw: 0
  };
  let cancels = 0;

  // Danh sách kịch bản xoay vòng kết quả RESULT
  const scenarios = [
    { outcome: 'player', player: 'rock', ai: 'scissors' },
    { outcome: 'ai', player: 'rock', ai: 'paper' },
    { outcome: 'draw', player: 'paper', ai: 'paper' },
    { outcome: 'unknown', player: null, ai: 'rock' }
  ];
  let scenarioIndex = 0;

  let renderCallback = null;
  let animFrameId = null;
  let currentStepIndex = 0;
  let stepStartTime = 0;
  let isRunning = false;

  function isDemoMode() {
    return isDemo;
  }

  /**
   * Tạo payload state khớp 100% snapshot của server.py
   */
  function buildState(overrides = {}) {
    return {
      status: 'ready',
      error: null,
      fps: CONFIG.fps,
      has_hand: true,
      a: { ...CONFIG.a },
      b: { ...CONFIG.b },
      box_r: CONFIG.box_r,
      band: CONFIG.band,
      gate_f: CONFIG.gate_f,
      a_hold: overrides.a_hold ?? 0.0,
      passed: overrides.passed ?? false,
      warn: overrides.warn ?? false,
      off_line: overrides.off_line ?? false,
      hand: overrides.hand !== undefined ? overrides.hand : null,
      phase: overrides.phase ?? 'WAIT',
      msg: overrides.msg ?? '',
      outcome: overrides.outcome ?? null,
      player: overrides.player ?? null,
      ai: overrides.ai ?? null,
      score: { ...score },
      cancels: cancels,
      ...overrides
    };
  }

  function emit(state) {
    if (typeof renderCallback === 'function') {
      renderCallback(state);
    }
  }

  /**
   * Định nghĩa các bước của kịch bản Demo:
   * duration (ms), generator: nhận elapsed (0 -> duration) và trả về overrides cho buildState()
   */
  const demoSteps = [
    // ========== VÁN 1: BÌNH THƯỜNG (BẠN THẮNG) ==========
    // Bước 0: WAIT 2 giây
    {
      duration: 2000,
      onEnter: () => {},
      generate: () => ({
        phase: 'WAIT',
        msg: 'Nhấn SPACE để chơi',
        hand: null,
        a_hold: 0,
        passed: false,
        warn: false,
        off_line: false
      })
    },
    // Bước 1: NEED_A 2 giây (a_hold chạy 0 -> 1, msg "Giữ BÚA trong ô A")
    {
      duration: 2000,
      onEnter: () => {},
      generate: (elapsed, dur) => {
        const prog = Math.min(1.0, elapsed / dur);
        return {
          phase: 'NEED_A',
          msg: 'Giữ BÚA trong ô A',
          hand: { x: 0.2, y: 0.5 },
          a_hold: prog,
          passed: false,
          warn: false,
          off_line: false
        };
      }
    },
    // Bước 2: GO 1 giây (passed=false, tay đi từ A tới gần vạch)
    {
      duration: 1000,
      onEnter: () => {},
      generate: (elapsed, dur) => {
        const prog = elapsed / dur;
        // Đi từ x=0.2 tới x=0.33 (vạch ở x=0.35)
        const x = 0.2 + (0.33 - 0.2) * prog;
        return {
          phase: 'GO',
          msg: '',
          hand: { x: Math.round(x * 1000) / 1000, y: 0.5 },
          a_hold: 1.0,
          passed: false,
          warn: false,
          off_line: false
        };
      }
    },
    // Bước 3: GO 1 giây (passed=true, tay đi tiếp tới gần B)
    {
      duration: 1000,
      onEnter: () => {},
      generate: (elapsed, dur) => {
        const prog = elapsed / dur;
        // Đi từ x=0.35 tới x=0.76 (B ở x=0.8)
        const x = 0.35 + (0.76 - 0.35) * prog;
        return {
          phase: 'GO',
          msg: '',
          hand: { x: Math.round(x * 1000) / 1000, y: 0.5 },
          a_hold: 1.0,
          passed: true,
          warn: false,
          off_line: false
        };
      }
    },
    // Bước 4: off_line=true 0,5 giây (tay chệch khỏi đường thẳng)
    {
      duration: 500,
      onEnter: () => {},
      generate: () => ({
        phase: 'GO',
        msg: '',
        hand: { x: 0.77, y: 0.32 }, // Lệch trục y ra ngoài dải 0.09
        a_hold: 1.0,
        passed: true,
        warn: false,
        off_line: true
      })
    },
    // Bước 5: CONFIRM 0,2 giây (tay chạm ô B)
    {
      duration: 200,
      onEnter: () => {},
      generate: () => ({
        phase: 'CONFIRM',
        msg: '',
        hand: { x: 0.8, y: 0.5 },
        a_hold: 1.0,
        passed: true,
        warn: false,
        off_line: false
      })
    },
    // Bước 6: RESULT 3 giây (Lộ đáp án, cộng điểm)
    {
      duration: 3000,
      onEnter: () => {
        const sc = scenarios[scenarioIndex % scenarios.length];
        scenarioIndex++;
        if (sc.outcome === 'player') score.player++;
        else if (sc.outcome === 'ai') score.ai++;
        else if (sc.outcome === 'draw') score.draw++;
      },
      generate: () => {
        const sc = scenarios[(scenarioIndex - 1 + scenarios.length) % scenarios.length];
        return {
          phase: 'RESULT',
          msg: 'Nhấn SPACE để chơi tiếp',
          hand: null,
          a_hold: 1.0,
          passed: true,
          warn: false,
          off_line: false,
          outcome: sc.outcome,
          player: sc.player,
          ai: sc.ai
        };
      }
    },

    // ========== VÁN 2: CẢNH BÁO & HỦY VÁN (CANCEL) ==========
    // Bước 7: NEED_A 1 giây
    {
      duration: 1000,
      onEnter: () => {},
      generate: (elapsed, dur) => {
        const prog = Math.min(1.0, elapsed / dur);
        return {
          phase: 'NEED_A',
          msg: 'Nắm tay thành BÚA, đưa vào ô A',
          hand: { x: 0.2, y: 0.5 },
          a_hold: prog,
          passed: false,
          warn: false,
          off_line: false
        };
      }
    },
    // Bước 8: GO với warn=true 0,5 giây (chữ cảnh báo: sai hình trước vạch)
    {
      duration: 500,
      onEnter: () => {},
      generate: () => ({
        phase: 'GO',
        msg: '',
        hand: { x: 0.26, y: 0.5 },
        a_hold: 1.0,
        passed: false,
        warn: true,
        off_line: false
      })
    },
    // Bước 9: NEED_A 1,5 giây với msg "Cảnh báo: phải giữ BÚA tới vạch! Đưa BÚA về ô A"
    {
      duration: 1500,
      onEnter: () => {},
      generate: (elapsed, dur) => {
        const prog = Math.min(1.0, elapsed / dur);
        return {
          phase: 'NEED_A',
          msg: 'Cảnh báo: phải giữ BÚA tới vạch! Đưa BÚA về ô A',
          hand: { x: 0.2, y: 0.5 },
          a_hold: prog,
          passed: false,
          warn: false,
          off_line: false
        };
      }
    },
    // Bước 10: GO 1 giây (làm lại thành công, qua vạch tới B)
    {
      duration: 1000,
      onEnter: () => {},
      generate: (elapsed, dur) => {
        const prog = elapsed / dur;
        const x = 0.2 + (0.8 - 0.2) * prog;
        const passed = x >= 0.35;
        return {
          phase: 'GO',
          msg: '',
          hand: { x: Math.round(x * 1000) / 1000, y: 0.5 },
          a_hold: 1.0,
          passed: passed,
          warn: false,
          off_line: false
        };
      }
    },
    // Bước 11: CONFIRM 0,2 giây
    {
      duration: 200,
      onEnter: () => {},
      generate: () => ({
        phase: 'CONFIRM',
        msg: '',
        hand: { x: 0.8, y: 0.5 },
        a_hold: 1.0,
        passed: true,
        warn: false,
        off_line: false
      })
    },
    // Bước 12: CANCEL 3 giây (Ván bị hủy: "Tay lúc chạm B khác tay sau 0,2 giây")
    {
      duration: 3000,
      onEnter: () => {
        cancels++;
      },
      generate: () => ({
        phase: 'CANCEL',
        msg: 'Tay lúc chạm B khác tay sau 0,2 giây',
        hand: null,
        a_hold: 1.0,
        passed: true,
        warn: false,
        off_line: false,
        outcome: null,
        player: null,
        ai: null
      })
    }
  ];

  /**
   * Vòng lặp tick animation mượt mà ~30fps
   */
  function loop(now) {
    if (!isRunning) return;

    if (!stepStartTime) {
      stepStartTime = now;
      demoSteps[currentStepIndex].onEnter();
    }

    const currentStep = demoSteps[currentStepIndex];
    const elapsed = now - stepStartTime;

    if (elapsed >= currentStep.duration) {
      // Chuyển sang bước tiếp theo
      currentStepIndex = (currentStepIndex + 1) % demoSteps.length;
      stepStartTime = now;
      demoSteps[currentStepIndex].onEnter();
    }

    const stateOverrides = demoSteps[currentStepIndex].generate(
      Math.min(currentStep.duration, now - stepStartTime),
      currentStep.duration
    );

    emit(buildState(stateOverrides));

    animFrameId = requestAnimationFrame(loop);
  }

  function start(onStateChange) {
    if (!isDemo) return;
    renderCallback = onStateChange;
    isRunning = true;
    currentStepIndex = 0;
    stepStartTime = 0;
    animFrameId = requestAnimationFrame(loop);
  }

  function stop() {
    isRunning = false;
    if (animFrameId) {
      cancelAnimationFrame(animFrameId);
      animFrameId = null;
    }
    renderCallback = null;
  }

  /**
   * Xử lý lệnh phím / nút bấm trong chế độ Demo
   */
  function handleCmd(cmdName) {
    if (!isDemo) return;

    if (cmdName === 'play') {
      const cur = demoSteps[currentStepIndex];
      const overrides = cur.generate(0, cur.duration);
      const phase = overrides.phase;

      // Nếu đang ở WAIT, RESULT hoặc CANCEL: nhảy ngay tới bước chơi tiếp theo
      if (phase === 'WAIT') {
        currentStepIndex = 1; // Nhảy vào NEED_A
        stepStartTime = performance.now();
        demoSteps[currentStepIndex].onEnter();
      } else if (phase === 'RESULT') {
        currentStepIndex = 7; // Nhảy sang ván 2
        stepStartTime = performance.now();
        demoSteps[currentStepIndex].onEnter();
      } else if (phase === 'CANCEL') {
        currentStepIndex = 0; // Quay về WAIT hoặc ván 1
        stepStartTime = performance.now();
        demoSteps[currentStepIndex].onEnter();
      }
    } else if (cmdName === 'reset') {
      // Đặt lại điểm số và số ván hủy
      score.player = 0;
      score.ai = 0;
      score.draw = 0;
      cancels = 0;
    }
  }

  return {
    isDemoMode,
    start,
    stop,
    handleCmd
  };
})();

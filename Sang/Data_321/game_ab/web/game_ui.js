/**
 * ==========================================================================
 * OẲN TÙ TÌ AI - LOGIC MÀN CHƠI (GAME UI)
 * 
 * Tuân thủ nghiêm ngặt hợp đồng với server.py:
 * - render(state): CHỈ VẼ theo state nhận được từ server / demo.
 * - TUYỆT ĐỐI không tự tính thắng/thua hay đoán cử chỉ ở JS, lấy nguyên từ server.
 * - Không có bất kỳ phần tử nào che đè lên webcam (ngoại trừ SVG overlay).
 * - Thanh hướng dẫn (hint-bar) nằm ngay dưới webcam, chiều cao cố định 76px.
 * - Dòng kết quả (ai-result-box) nằm dưới ô tay AI, chiều cao cố định 76px.
 * - Tối ưu FPS: Dựng sẵn DOM & SVG 1 lần, chỉ đổi thuộc tính khi giá trị thay đổi.
 * ==========================================================================
 */

const GameUI = (() => {
  // Trạng thái vòng đời màn chơi
  let isRunning = false;
  let isPolling = false;
  let pollTimeout = null;
  let keydownHandler = null;

  // Bản đồ tên tiếng Việt của cử chỉ
  const GESTURE_NAMES = {
    paper: 'Bao',
    rock: 'Búa',
    scissors: 'Kéo'
  };

  // Cache tham chiếu DOM & SVG (Khởi tạo MỘT LẦN duy nhất)
  let dom = null;

  // Cache giá trị trước để so sánh, CHỈ cập nhật DOM khi giá trị thay đổi
  const prev = {
    status: null,
    fpsText: '',
    fpsLastTime: 0,
    scorePlayer: null,
    scoreAi: null,
    scoreDraw: null,
    scoreCancels: null,
    // Overlay coordinates
    ax: null, ay: null, bx: null, by: null,
    boxR: null, bandW: null, gateF: null,
    aHold: null,
    passed: null,
    warn: null,
    offLine: null,
    gateDisplay: null,
    gateStroke: null,
    circleAStroke: null,
    subtextAFill: null,
    circleBStroke: null,
    circleBOpacity: null,
    textBFill: null,
    textBOpacity: null,
    lineStroke: null,
    lineOpacity: null,
    bandStroke: null,
    handX: null, handY: null, handVisible: null,
    // Hint Bar & AI Result
    hintKey: '',
    aiResultKey: '',
    aiVisibleItem: null,
    // Play button
    btnDisabled: null,
    btnLabel: '',
    // Audio trigger tracking
    phase: null
  };

  /**
   * Lấy và lưu cache các phần tử DOM/SVG một lần duy nhất
   */
  function getDOM() {
    if (!dom) {
      dom = {
        // Webcam & Video
        camFeed: document.getElementById('cam-feed'),
        camPlaceholder: document.getElementById('cam-placeholder'),
        camDemoBadge: document.getElementById('cam-demo-badge'),

        // Header, Status & Scoreboard
        statusPill: document.getElementById('status-pill'),
        aiStatusText: document.getElementById('ai-status-text'),
        fpsCounter: document.getElementById('fps-counter'),
        scorePlayer: document.getElementById('score-player'),
        scoreAi: document.getElementById('score-ai'),
        scoreDraw: document.getElementById('score-draw'),
        scoreCancels: document.getElementById('score-cancels'),

        // Overlay SVG Elements
        overlay: document.getElementById('overlay'),
        ovBand: document.getElementById('ov-band'),
        ovLine: document.getElementById('ov-line'),
        ovGate: document.getElementById('ov-gate'),
        ovCircleB: document.getElementById('ov-circle-b'),
        ovTextB: document.getElementById('ov-text-b'),
        ovCircleA: document.getElementById('ov-circle-a'),
        ovProgressA: document.getElementById('ov-progress-a'),
        ovTextA: document.getElementById('ov-text-a'),
        ovSubtextA: document.getElementById('ov-subtext-a'),
        ovHand: document.getElementById('ov-hand'),

        // Thanh hướng dẫn nằm ngay dưới webcam
        hintBar: document.getElementById('hint-bar'),
        hintTitle: document.getElementById('hint-title'),
        hintDetail: document.getElementById('hint-detail'),

        // AI Viewport, Hand Box & Result Box
        aiHandBox: document.getElementById('ai-hand-box'),
        aiHandMystery: document.getElementById('ai-hand-mystery'),
        aiHandRock: document.getElementById('ai-hand-rock'),
        aiHandPaper: document.getElementById('ai-hand-paper'),
        aiHandScissors: document.getElementById('ai-hand-scissors'),
        aiResultBox: document.getElementById('ai-result-box'),
        aiResultTitle: document.getElementById('ai-result-title'),
        aiResultDetail: document.getElementById('ai-result-detail'),

        // Controls
        btnGamePlay: document.getElementById('btn-game-play'),
        btnGameReset: document.getElementById('btn-game-reset')
      };
    }
    return dom;
  }

  /**
   * Phát âm thanh tổng hợp qua GameApp
   */
  function playSound(type) {
    if (typeof GameApp !== 'undefined' && typeof GameApp.playSynthSound === 'function') {
      GameApp.playSynthSound(type);
    }
  }

  /**
   * Gửi lệnh tới server hoặc bộ phát Demo
   */
  async function sendCommand(cmdName) {
    if (typeof DemoState !== 'undefined' && DemoState.isDemoMode()) {
      DemoState.handleCmd(cmdName);
      return;
    }
    try {
      await fetch(`/cmd/${cmdName}`, { method: 'POST' });
    } catch (err) {
      console.warn(`[GameUI] Gửi lệnh /cmd/${cmdName} thất bại:`, err);
    }
  }

  /**
   * Cập nhật Bảng điểm chỉ khi số điểm thay đổi
   */
  function updateScoreboard(d, state) {
    const score = state.score;
    if (!score) return;

    if (score.player !== prev.scorePlayer) {
      d.scorePlayer.textContent = score.player ?? 0;
      prev.scorePlayer = score.player;
    }
    if (score.ai !== prev.scoreAi) {
      d.scoreAi.textContent = score.ai ?? 0;
      prev.scoreAi = score.ai;
    }
    if (score.draw !== prev.scoreDraw) {
      d.scoreDraw.textContent = score.draw ?? 0;
      prev.scoreDraw = score.draw;
    }
    const cancels = state.cancels ?? 0;
    if (cancels !== prev.scoreCancels) {
      d.scoreCancels.textContent = cancels;
      prev.scoreCancels = cancels;
    }
  }

  /**
   * Cập nhật FPS tối đa 1 lần/giây để không tốn tài nguyên
   */
  function updateFps(d, fpsValue) {
    const now = performance.now();
    if (now - prev.fpsLastTime >= 1000) {
      const rounded = Math.round(fpsValue || 0);
      const text = `${rounded} FPS`;
      if (text !== prev.fpsText) {
        d.fpsCounter.textContent = text;
        prev.fpsText = text;
      }
      prev.fpsLastTime = now;
    }
  }

  /**
   * Cập nhật Overlay SVG theo toạ độ a, b, box_r, band, gate_f, hand
   * CHỈ gán thuộc tính khi giá trị khác lần trước
   */
  function updateOverlay(d, state) {
    if (!state.a || !state.b) return;

    const ax = state.a.x * 640;
    const ay = state.a.y * 480;
    const bx = state.b.x * 640;
    const by = state.b.y * 480;
    const boxR = (state.box_r ?? 0.13) * 480;
    const bandW = (state.band ?? 0.09) * 480;
    const gateF = state.gate_f ?? 0.25;
    const aHold = Math.max(0, Math.min(1, state.a_hold || 0));

    // 1. Cập nhật vị trí hình học cơ bản nếu a, b thay đổi
    if (ax !== prev.ax || ay !== prev.ay || bx !== prev.bx || by !== prev.by ||
        boxR !== prev.boxR || bandW !== prev.bandW || gateF !== prev.gateF) {
      
      // Dải cho phép
      d.ovBand.setAttribute('x1', ax);
      d.ovBand.setAttribute('y1', ay);
      d.ovBand.setAttribute('x2', bx);
      d.ovBand.setAttribute('y2', by);
      d.ovBand.setAttribute('stroke-width', bandW * 2);

      // Đường nối A-B
      d.ovLine.setAttribute('x1', ax);
      d.ovLine.setAttribute('y1', ay);
      d.ovLine.setAttribute('x2', bx);
      d.ovLine.setAttribute('y2', by);

      // Ô B
      d.ovCircleB.setAttribute('cx', bx);
      d.ovCircleB.setAttribute('cy', by);
      d.ovCircleB.setAttribute('r', boxR);
      d.ovTextB.setAttribute('x', bx);
      d.ovTextB.setAttribute('y', by + 9);

      // Ô A
      d.ovCircleA.setAttribute('cx', ax);
      d.ovCircleA.setAttribute('cy', ay);
      d.ovCircleA.setAttribute('r', boxR);

      d.ovProgressA.setAttribute('cx', ax);
      d.ovProgressA.setAttribute('cy', ay);
      d.ovProgressA.setAttribute('r', boxR);
      d.ovProgressA.setAttribute('transform', `rotate(-90 ${ax} ${ay})`);
      const circ = 2 * Math.PI * boxR;
      d.ovProgressA.setAttribute('stroke-dasharray', circ);

      d.ovTextA.setAttribute('x', ax);
      d.ovTextA.setAttribute('y', ay + 2);
      d.ovSubtextA.setAttribute('x', ax);
      d.ovSubtextA.setAttribute('y', ay + 22);

      // Vạch gate_f (đoạn thẳng vuông góc đường A-B)
      const dx = bx - ax;
      const dy = by - ay;
      const len = Math.hypot(dx, dy) || 1;
      const nx = -dy / len;
      const ny = dx / len;
      const gx = ax + dx * gateF;
      const gy = ay + dy * gateF;
      const hGate = 1.3 * bandW;
      d.ovGate.setAttribute('x1', gx + nx * hGate);
      d.ovGate.setAttribute('y1', gy + ny * hGate);
      d.ovGate.setAttribute('x2', gx - nx * hGate);
      d.ovGate.setAttribute('y2', gy - ny * hGate);

      prev.ax = ax; prev.ay = ay; prev.bx = bx; prev.by = by;
      prev.boxR = boxR; prev.bandW = bandW; prev.gateF = gateF;
    }

    // 2. Vòng tiến độ giữ búa theo a_hold
    if (aHold !== prev.aHold) {
      const circ = 2 * Math.PI * (prev.boxR || boxR);
      d.ovProgressA.setAttribute('stroke-dashoffset', circ * (1 - aHold));
      prev.aHold = aHold;
    }

    // 3. Hiển thị & màu sắc theo phase
    const phase = state.phase;
    const passed = !!state.passed;
    const offLine = !!state.off_line;

    // Vạch gate chỉ hiện ở NEED_A và GO
    const showGate = (phase === 'NEED_A' || phase === 'GO');
    if (showGate !== prev.gateDisplay) {
      d.ovGate.style.display = showGate ? 'block' : 'none';
      prev.gateDisplay = showGate;
    }

    if (phase === 'NEED_A') {
      const gateColor = '#f59e0b';
      if (gateColor !== prev.gateStroke) {
        d.ovGate.setAttribute('stroke', gateColor);
        prev.gateStroke = gateColor;
      }

      const aColor = aHold > 0 ? '#22c55e' : '#f59e0b';
      if (aColor !== prev.circleAStroke) {
        d.ovCircleA.setAttribute('stroke', aColor);
        d.ovSubtextA.setAttribute('fill', aColor);
        prev.circleAStroke = aColor;
        prev.subtextAFill = aColor;
      }

      // Làm Ô B rõ hơn: ở NEED_A, ô B và đường nối màu xám sáng (#94a3b8, opacity 0.8)
      if ('#94a3b8' !== prev.circleBStroke || prev.circleBOpacity !== '0.8') {
        d.ovCircleB.setAttribute('stroke', '#94a3b8');
        d.ovCircleB.setAttribute('stroke-opacity', '0.8');
        d.ovTextB.setAttribute('fill', '#94a3b8');
        d.ovTextB.setAttribute('fill-opacity', '0.8');
        d.ovLine.setAttribute('stroke', '#94a3b8');
        d.ovLine.setAttribute('stroke-opacity', '0.8');
        d.ovBand.setAttribute('stroke', '#ffffff');
        prev.circleBStroke = '#94a3b8';
        prev.circleBOpacity = '0.8';
        prev.textBFill = '#94a3b8';
        prev.textBOpacity = '0.8';
        prev.lineStroke = '#94a3b8';
        prev.lineOpacity = '0.8';
        prev.bandStroke = '#ffffff';
      }
    } else if (phase === 'GO') {
      const gateColor = passed ? '#22c55e' : '#f97316';
      if (gateColor !== prev.gateStroke) {
        d.ovGate.setAttribute('stroke', gateColor);
        prev.gateStroke = gateColor;
      }

      if ('#22c55e' !== prev.circleAStroke) {
        d.ovCircleA.setAttribute('stroke', '#22c55e');
        d.ovSubtextA.setAttribute('fill', '#22c55e');
        prev.circleAStroke = '#22c55e';
        prev.subtextAFill = '#22c55e';
      }

      const lineCol = offLine ? '#ef4444' : '#3b82f6';
      const bandCol = offLine ? '#ef4444' : '#ffffff';
      if (lineCol !== prev.lineStroke || prev.lineOpacity !== null) {
        d.ovLine.setAttribute('stroke', lineCol);
        d.ovLine.removeAttribute('stroke-opacity');
        prev.lineStroke = lineCol;
        prev.lineOpacity = null;
      }
      if (bandCol !== prev.bandStroke) {
        d.ovBand.setAttribute('stroke', bandCol);
        prev.bandStroke = bandCol;
      }

      if ('#3b82f6' !== prev.circleBStroke || prev.circleBOpacity !== null) {
        d.ovCircleB.setAttribute('stroke', '#3b82f6');
        d.ovCircleB.removeAttribute('stroke-opacity');
        d.ovTextB.setAttribute('fill', '#f1f5f9');
        d.ovTextB.removeAttribute('fill-opacity');
        prev.circleBStroke = '#3b82f6';
        prev.circleBOpacity = null;
        prev.textBFill = '#f1f5f9';
        prev.textBOpacity = null;
      }
    }

    // 4. Chấm tròn vị trí bàn tay
    const hand = state.hand;
    const showHand = (hand !== null && phase !== 'RESULT' && phase !== 'CANCEL');
    if (showHand !== prev.handVisible) {
      d.ovHand.style.display = showHand ? 'block' : 'none';
      prev.handVisible = showHand;
    }
    if (showHand && hand) {
      const hx = hand.x * 640;
      const hy = hand.y * 480;
      if (hx !== prev.handX || hy !== prev.handY) {
        d.ovHand.setAttribute('cx', hx);
        d.ovHand.setAttribute('cy', hy);
        prev.handX = hx;
        prev.handY = hy;
      }
    }
  }

  /**
   * Cập nhật Thanh hướng dẫn nằm ngay dưới webcam (id="hint-bar")
   * - CONFIRM: giữ NGUYÊN nội dung đang hiện (không đổi, không nhấp nháy).
   * - Các phase khác cập nhật tiêu đề, chi tiết và màu viền trái theo hợp đồng.
   */
  function updateHintBar(d, state) {
    const phase = state.phase;

    // CONFIRM: giữ NGUYÊN nội dung đang hiện
    if (phase === 'CONFIRM') {
      return;
    }

    let title = '';
    let detail = '';
    let borderClass = 'border-player';

    switch (phase) {
      case 'WAIT':
        title = 'SẴN SÀNG';
        detail = state.msg || 'Nhấn SPACE để chơi';
        borderClass = 'border-player';
        break;

      case 'NEED_A':
        title = 'VÀO Ô A';
        detail = state.msg || 'Nắm tay thành BÚA, đưa vào ô A';
        borderClass = 'border-warn';
        break;

      case 'GO':
        if (state.off_line) {
          title = 'LỆCH ĐƯỜNG!';
          detail = 'Đưa tay về đường nối A–B';
          borderClass = 'border-bad';
        } else if (state.warn) {
          title = 'CẢNH BÁO: GIỮ BÚA TỚI VẠCH!';
          detail = 'Đổi tay trước vạch sẽ phải làm lại';
          borderClass = 'border-bad';
        } else if (!state.passed) {
          title = 'GIỮ BÚA TỚI VẠCH';
          detail = 'Giữ nắm tay đến khi chạm vạch';
          borderClass = 'border-warn';
        } else {
          title = 'ĐI TỚI Ô B!';
          detail = 'Chạm ô B để AI ra quân';
          borderClass = 'border-player';
        }
        break;

      case 'RESULT':
        title = 'NHẤN SPACE ĐỂ CHƠI TIẾP';
        detail = '';
        borderClass = 'border-ok';
        break;

      case 'CANCEL':
        title = 'VÁN BỊ HỦY';
        detail = `${state.msg || 'Tay lúc chạm B khác tay sau 0,2 giây'} • Nhấn SPACE để chơi lại`;
        borderClass = 'border-muted';
        break;

      default:
        title = 'SẴN SÀNG';
        detail = state.msg || '';
        borderClass = 'border-player';
        break;
    }

    const key = `${title}|${detail}|${borderClass}`;
    if (key !== prev.hintKey) {
      d.hintTitle.textContent = title;
      d.hintDetail.textContent = detail;
      d.hintBar.className = `hint-bar ${borderClass}`;
      prev.hintKey = key;
    }
  }

  /**
   * Cập nhật Khung kết quả ở khung AI (dưới ô tay AI)
   * - Mọi phase trừ RESULT và CANCEL: "AI ĐANG SUY NGHĨ" (ở WAIT là "AI SẴN SÀNG").
   * - RESULT: Dòng lớn kết quả ("AI THẮNG" / "BẠN THẮNG" / "HÒA" / "KHÔNG XÁC ĐỊNH"), dòng nhỏ: "Bạn ra: ... • AI ra: ...".
   * - CANCEL: "VÁN BỊ HỦY" kèm "Ván này không tính điểm".
   */
  function updateAiResult(d, state) {
    const phase = state.phase;
    let title = '';
    let detail = '';
    let colorClass = 'color-muted';

    if (phase === 'RESULT') {
      const outcome = state.outcome;
      if (outcome === 'ai') {
        title = 'AI THẮNG';
        colorClass = 'color-ai';
      } else if (outcome === 'player') {
        title = 'BẠN THẮNG';
        colorClass = 'color-ok';
      } else if (outcome === 'draw') {
        title = 'HÒA';
        colorClass = 'color-warn';
      } else {
        title = 'KHÔNG XÁC ĐỊNH';
        colorClass = 'color-muted';
      }

      if (outcome === 'unknown') {
        detail = 'Ván này không tính điểm';
      } else {
        const pName = GESTURE_NAMES[state.player] || '—';
        const aiName = GESTURE_NAMES[state.ai] || '—';
        detail = `Bạn ra: ${pName} • AI ra: ${aiName}`;
      }
    } else if (phase === 'CANCEL') {
      title = 'VÁN BỊ HỦY';
      detail = 'Ván này không tính điểm';
      colorClass = 'color-muted';
    } else if (phase === 'WAIT') {
      title = 'AI SẴN SÀNG';
      detail = '';
      colorClass = 'color-muted';
    } else {
      // NEED_A, GO, CONFIRM
      title = 'AI ĐANG SUY NGHĨ';
      detail = '';
      colorClass = 'color-muted';
    }

    const key = `${title}|${detail}|${colorClass}`;
    if (key !== prev.aiResultKey) {
      d.aiResultTitle.textContent = title;
      d.aiResultDetail.textContent = detail;
      d.aiResultTitle.className = `ai-result-title ${colorClass}`;
      prev.aiResultKey = key;
    }
  }

  /**
   * Cập nhật Ô tay AI
   * - Mọi phase trừ RESULT: hiện tay bí ẩn + "?"
   * - RESULT: hiện hình bàn tay AI ra (rock, paper, scissors) kèm animation nhẹ
   */
  function updateAiHand(d, state) {
    let target = 'mystery';

    if (state.phase === 'RESULT' && state.ai) {
      target = state.ai; // 'rock' | 'paper' | 'scissors'
    }

    if (target !== prev.aiVisibleItem) {
      // Ẩn tất cả
      d.aiHandMystery.classList.remove('active', 'revealed');
      d.aiHandRock.classList.remove('active', 'revealed');
      d.aiHandPaper.classList.remove('active', 'revealed');
      d.aiHandScissors.classList.remove('active', 'revealed');

      // Bật mục tương ứng
      let el = null;
      if (target === 'rock') el = d.aiHandRock;
      else if (target === 'paper') el = d.aiHandPaper;
      else if (target === 'scissors') el = d.aiHandScissors;
      else el = d.aiHandMystery;

      if (el) {
        el.classList.add('active');
        if (target !== 'mystery') {
          el.classList.add('revealed');
        }
      }
      prev.aiVisibleItem = target;
    }
  }

  /**
   * Cập nhật trạng thái và chữ trên nút CHƠI (SPACE)
   */
  function updatePlayButton(d, state) {
    const phase = state.phase;
    const canPlay = (phase === 'WAIT' || phase === 'RESULT' || phase === 'CANCEL');
    const label = (phase === 'RESULT' || phase === 'CANCEL') ? 'CHƠI TIẾP (SPACE)' : 'CHƠI (SPACE)';

    if (canPlay === prev.btnDisabled) { // canPlay = true -> disabled = false
      d.btnGamePlay.disabled = !canPlay;
      prev.btnDisabled = !canPlay;
    }

    if (label !== prev.btnLabel) {
      d.btnGamePlay.textContent = label;
      prev.btnLabel = label;
    }
  }

  /**
   * Kiểm tra và phát âm thanh khi đổi phase
   * - NEED_A -> GO: playSynthSound('click')
   * - GO -> CONFIRM: playSynthSound('start')
   */
  function checkAudioTransitions(newPhase) {
    if (prev.phase !== newPhase) {
      if (prev.phase === 'NEED_A' && newPhase === 'GO') {
        playSound('click');
      } else if (prev.phase === 'GO' && newPhase === 'CONFIRM') {
        playSound('start');
      }
      prev.phase = newPhase;
    }
  }

  /**
   * HÀM RENDER CHÍNH (Được gọi mỗi khi nhận được state mới)
   * TUYỆT ĐỐI không tính toán hay đoán cử chỉ, chỉ vẽ theo dữ liệu.
   */
  function render(state) {
    const d = getDOM();
    if (!d || !d.hintBar) return;

    // 1. Kiểm tra trạng thái máy chủ (disconnected, loading, error)
    if (state.status === 'disconnected') {
      if (prev.status !== 'disconnected') {
        d.aiStatusText.textContent = 'Mất kết nối server';
        d.statusPill.className = 'status-pill disconnected';

        d.hintTitle.textContent = 'MẤT KẾT NỐI';
        d.hintDetail.textContent = 'Chạy server.py rồi mở http://127.0.0.1:8000';
        d.hintBar.className = 'hint-bar border-bad';

        d.aiResultTitle.textContent = 'MẤT KẾT NỐI';
        d.aiResultDetail.textContent = '';
        d.aiResultTitle.className = 'ai-result-title color-bad';

        d.btnGamePlay.disabled = true;
        prev.btnDisabled = true;
        prev.status = 'disconnected';
      }
      return;
    }

    if (state.status === 'loading') {
      if (prev.status !== 'loading') {
        d.aiStatusText.textContent = 'Đang khởi động';
        d.statusPill.className = 'status-pill loading';

        d.hintTitle.textContent = 'ĐANG KHỞI ĐỘNG';
        d.hintDetail.textContent = 'Đang tải model và camera...';
        d.hintBar.className = 'hint-bar border-muted';

        d.aiResultTitle.textContent = 'ĐANG KHỞI ĐỘNG';
        d.aiResultDetail.textContent = '';
        d.aiResultTitle.className = 'ai-result-title color-muted';

        d.btnGamePlay.disabled = true;
        prev.btnDisabled = true;
        prev.status = 'loading';
      }
      return;
    }

    if (state.status === 'error') {
      if (prev.status !== 'error') {
        d.aiStatusText.textContent = 'Lỗi';
        d.statusPill.className = 'status-pill error';

        d.hintTitle.textContent = 'LỖI';
        d.hintDetail.textContent = `${state.error || 'Đã xảy ra lỗi server.'} Kiểm tra server rồi tải lại trang (F5)`;
        d.hintBar.className = 'hint-bar border-bad';

        d.aiResultTitle.textContent = 'LỖI';
        d.aiResultDetail.textContent = '';
        d.aiResultTitle.className = 'ai-result-title color-bad';

        d.btnGamePlay.disabled = true;
        prev.btnDisabled = true;
        prev.status = 'error';
      }
      return;
    }

    // status === 'ready'
    if (prev.status !== 'ready') {
      d.aiStatusText.textContent = 'Sẵn sàng';
      d.statusPill.className = 'status-pill ready';
      prev.status = 'ready';
    }

    // 2. Kích hoạt âm thanh chuyển tiếp phase
    checkAudioTransitions(state.phase);

    // 3. Cập nhật FPS (tối đa 1 lần/giây)
    updateFps(d, state.fps);

    // 4. Cập nhật Bảng điểm
    updateScoreboard(d, state);

    // 5. Cập nhật Overlay SVG
    updateOverlay(d, state);

    // 6. Cập nhật Thanh hướng dẫn dưới webcam
    updateHintBar(d, state);

    // 7. Cập nhật Ô tay AI & Kết quả khung AI
    updateAiHand(d, state);
    updateAiResult(d, state);

    // 8. Cập nhật Nút CHƠI (SPACE)
    updatePlayButton(d, state);
  }

  /**
   * Vòng lặp fetch('/state') dùng setTimeout nối tiếp (không dùng setInterval)
   * Đợi render xong, nghỉ 33ms rồi mới gọi lần kế
   */
  async function pollServer() {
    if (!isRunning || !isPolling) return;

    try {
      const res = await fetch('/state', { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      render(data);

      if (isRunning && isPolling) {
        pollTimeout = setTimeout(pollServer, 33);
      }
    } catch (err) {
      render({ status: 'disconnected' });
      if (isRunning && isPolling) {
        pollTimeout = setTimeout(pollServer, 1000); // Thử lại sau 1 giây
      }
    }
  }

  function startPolling() {
    if (!isPolling) {
      isPolling = true;
      pollServer();
    }
  }

  function stopPolling() {
    isPolling = false;
    clearTimeout(pollTimeout);
    pollTimeout = null;
  }

  /**
   * Đăng ký phím điều khiển (chỉ nghe khi đang ở màn chơi)
   */
  function setupKeyboard() {
    keydownHandler = (e) => {
      if (!isRunning) return;

      // Phím SPACE: Chơi / chơi tiếp (chặn cuộn trang)
      if (e.code === 'Space') {
        e.preventDefault();
        const d = getDOM();
        if (d.btnGamePlay && !d.btnGamePlay.disabled) {
          sendCommand('play');
        }
      }

      // Phím R: Reset điểm
      if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        sendCommand('reset');
      }
    };
    window.addEventListener('keydown', keydownHandler);
  }

  /**
   * Đăng ký sự kiện click cho các nút
   */
  function setupButtons() {
    const d = getDOM();
    if (d.btnGamePlay) {
      d.btnGamePlay.onclick = () => {
        if (!d.btnGamePlay.disabled) {
          sendCommand('play');
        }
      };
    }
    if (d.btnGameReset) {
      d.btnGameReset.onclick = () => sendCommand('reset');
    }
  }

  /**
   * Xử lý khi tab trình duyệt bị ẩn hoặc hiển thị lại
   */
  function handleVisibilityChange() {
    if (document.hidden) {
      stopPolling();
    } else if (isRunning) {
      startPolling();
    }
  }

  /**
   * Kích hoạt màn chơi khi vào từ trang chủ
   */
  function start() {
    isRunning = true;
    prev.phase = null;
    prev.hintKey = '';
    prev.aiResultKey = '';
    prev.aiVisibleItem = null;

    const d = getDOM();
    setupKeyboard();
    setupButtons();
    document.addEventListener('visibilitychange', handleVisibilityChange);

    // Kiểm tra chế độ Demo (?demo=1)
    if (typeof DemoState !== 'undefined' && DemoState.isDemoMode()) {
      if (d.camDemoBadge) d.camDemoBadge.style.display = 'block';
      if (d.camPlaceholder) d.camPlaceholder.style.display = 'none';
      if (d.camFeed) d.camFeed.style.display = 'none';
      DemoState.start(render);
      return;
    }

    // Chế độ thật kết nối server Python
    if (d.camDemoBadge) d.camDemoBadge.style.display = 'none';
    if (d.camFeed) {
      d.camFeed.style.display = 'block';
      d.camFeed.src = '/video'; // Luồng MJPEG từ server
      d.camFeed.onload = () => {
        if (d.camPlaceholder) d.camPlaceholder.style.display = 'none';
      };
      d.camFeed.onerror = () => {
        if (d.camPlaceholder) d.camPlaceholder.style.display = 'flex';
      };
    }

    startPolling();
  }

  /**
   * Dừng màn chơi khi quay về trang chủ
   */
  function stop() {
    isRunning = false;
    stopPolling();

    if (keydownHandler) {
      window.removeEventListener('keydown', keydownHandler);
      keydownHandler = null;
    }
    document.removeEventListener('visibilitychange', handleVisibilityChange);

    const d = getDOM();
    if (d && d.camFeed) {
      d.camFeed.src = ''; // Giải phóng luồng video camera
    }

    if (typeof DemoState !== 'undefined' && DemoState.isDemoMode()) {
      DemoState.stop();
    }
  }

  return {
    start,
    stop,
    render
  };
})();

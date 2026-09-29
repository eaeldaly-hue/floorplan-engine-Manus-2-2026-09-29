(() => {
  const form = document.getElementById('upload-form');
  const fileInput = document.getElementById('plan-file');
  const dropzone = document.getElementById('dropzone');
  const analyzeButton = document.getElementById('analyze-button');
  const sampleButton = document.getElementById('sample-button');
  const errorBox = document.getElementById('error-box');
  const progressBox = document.getElementById('progress-box');
  const results = document.getElementById('results');
  const allowed = /\.(png|jpe?g|bmp|tiff?|pdf)$/i;
  let selectedFile = null;

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  function chooseFile(file) {
    errorBox.hidden = true;
    if (!file) return;
    if (!allowed.test(file.name)) {
      selectedFile = null;
      analyzeButton.disabled = true;
      errorBox.textContent = 'نوع الملف غير مدعوم. اختر PNG أو JPG أو TIFF أو PDF.';
      errorBox.hidden = false;
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      selectedFile = null;
      analyzeButton.disabled = true;
      errorBox.textContent = 'حجم الملف أكبر من 20 ميغابايت.';
      errorBox.hidden = false;
      return;
    }
    selectedFile = file;
    document.getElementById('file-title').textContent = 'الملف جاهز للتحليل';
    document.getElementById('file-subtitle').textContent = 'اضغط زر «حلّل المخطط» للبدء';
    document.getElementById('file-name').textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
    analyzeButton.disabled = false;
  }

  fileInput.addEventListener('change', () => chooseFile(fileInput.files[0]));
  for (const eventName of ['dragenter', 'dragover']) {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add('is-over');
    });
  }
  for (const eventName of ['dragleave', 'drop']) {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove('is-over');
    });
  }
  dropzone.addEventListener('drop', (event) => chooseFile(event.dataTransfer.files[0]));

  document.getElementById('new-analysis').addEventListener('click', () => {
    results.hidden = true;
    form.reset();
    selectedFile = null;
    analyzeButton.disabled = true;
    document.getElementById('file-title').textContent = 'اسحب الملف هنا أو اختره من جهازك';
    document.getElementById('file-subtitle').textContent = 'يفضّل مخططًا واضحًا ومستقيمًا مع ظهور النص والأبعاد';
    document.getElementById('file-name').textContent = '';
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  function renderArea(area) {
    if (!area || area.value === null || area.value === undefined) return 'المساحة غير متاحة';
    const value = Number(area.value);
    return `${value.toLocaleString('ar', { maximumFractionDigits: 2 })} ${area.unit}`;
  }

  function renderBoundary(boundary) {
    if (!boundary) return { label: 'لم تُرسم حدود موثوقة', confidence: 'غير متاح' };
    const source = boundary.method === 'wall-region' ? 'حدود من الجدران' :
      boundary.method === 'independent-wall-region' ? 'حدود من كاشف جدران إضافي' :
      boundary.method === 'dimension-only-estimate' ? 'حدود تقديرية من الأبعاد المطبوعة' :
      boundary.method === 'wall-ray-estimate' ? 'تقدير من امتداد الجدران' : 'مستنتجة من الأبعاد';
    return { label: source, confidence: `${Math.round((boundary.confidence || 0) * 100)}%` };
  }

  function roomCard(room, index, compact = false) {
    const card = el('article', `room-card${compact ? ' compact' : ''}`);
    const heading = el('div', 'room-card-head');
    const title = el('h4', 'room-name', `${String(index + 1).padStart(2, '0')} · ${room.name || 'مساحة غير مسماة'}`);
    const area = el('span', 'room-area', renderArea(room.area));
    heading.append(title, area);
    card.append(heading);

    const detail = el('div', 'room-details');
    const dims = room.dimensions?.display || room.dimensions?.text;
    if (dims) detail.append(el('span', '', `الأبعاد: ${dims}`));
    if (room.dimensions?.ocr_confidence !== undefined) {
      detail.append(el('span', '', `ثقة قراءة الأبعاد: ${Math.round(room.dimensions.ocr_confidence)}%`));
    }
    if (room.label_confidence !== undefined) {
      detail.append(el('span', '', `ثقة قراءة الاسم: ${Math.round(room.label_confidence)}%`));
    }
    const areaSource = room.area?.source === 'printed-dimensions' ? 'المساحة من الأبعاد المطبوعة' :
      room.area?.source === 'pixel-scale-estimate' ? 'المساحة تقديرية من مقياس الصورة' :
      room.area?.source === 'polygon-pixels' ? 'المساحة بالبكسل فقط' : '';
    if (areaSource) detail.append(el('span', '', areaSource));
    if (room.boundary?.bbox) {
      const box = room.boundary.bbox;
      detail.append(el('span', '', `صندوق الحدود: x=${box.x}, y=${box.y} · ${box.width} × ${box.height} px`));
    }
    const boundary = renderBoundary(room.boundary);
    detail.append(el('span', 'boundary-label', `${boundary.label} · الثقة ${boundary.confidence}`));
    card.append(detail);
    return card;
  }

  function renderResult(data) {
    document.getElementById('source-name').textContent = data.source_name || '';
    document.getElementById('room-count').textContent = `${data.room_count} غرفة`;
    document.getElementById('rooms-subtitle').textContent = 'كل بطاقة مرتبطة باسم/أبعاد مقروءة من المخطط أو بقياس هندسي تقديري.';

    const summary = document.getElementById('summary-grid');
    summary.replaceChildren();
    const scaleText = data.pixel_scale ? `${data.pixel_scale.pixels_per_unit} px/${data.pixel_scale.unit}` : 'لا يوجد مقياس موثوق';
    const summaries = [
      ['الغرف المسماة', data.room_count],
      ['مساحات غير مسماة', data.unlabeled_space_count],
      ['دقة الصورة', `${data.image.width} × ${data.image.height} px`],
      ['مقياس الرسم المستنتج', scaleText],
    ];
    summaries.forEach(([label, value]) => {
      const box = el('div', 'summary-card');
      box.append(el('span', 'summary-label', label), el('strong', 'summary-value', String(value)));
      summary.append(box);
    });

    const overlay = document.getElementById('overlay-image');
    overlay.src = `${data.overlay_url}?v=${Date.now()}`;
    document.getElementById('download-overlay').href = overlay.src;
    const jsonLink = document.getElementById('download-json');
    if (window.currentFloorplanJsonUrl) URL.revokeObjectURL(window.currentFloorplanJsonUrl);
    window.currentFloorplanJsonUrl = URL.createObjectURL(
      new Blob([JSON.stringify(data, null, 2)], { type: 'application/json;charset=utf-8' })
    );
    jsonLink.href = window.currentFloorplanJsonUrl;

    const list = document.getElementById('room-list');
    list.replaceChildren();
    (data.rooms || []).forEach((room, index) => list.append(roomCard(room, index)));

    const warnings = document.getElementById('warnings');
    warnings.replaceChildren();
    if (data.warnings?.length) {
      warnings.hidden = false;
      warnings.append(el('strong', '', 'ملاحظات على النتيجة'));
      const ul = el('ul');
      data.warnings.forEach((warning) => ul.append(el('li', '', warning)));
      warnings.append(ul);
    } else {
      warnings.hidden = true;
    }

    const unlabeledPanel = document.getElementById('unlabeled-panel');
    const unlabeledList = document.getElementById('unlabeled-list');
    unlabeledList.replaceChildren();
    if (data.unlabeled_spaces?.length) {
      unlabeledPanel.hidden = false;
      data.unlabeled_spaces.forEach((room, index) => unlabeledList.append(roomCard(room, index, true)));
    } else {
      unlabeledPanel.hidden = true;
    }

    const openingsPanel = document.getElementById('openings-panel');
    const openingList = document.getElementById('opening-list');
    const openingSummary = document.getElementById('opening-summary');
    openingList.replaceChildren();
    openingSummary.replaceChildren();
    const openings = data.openings || [];
    openingsPanel.hidden = openings.length === 0;
    if (openings.length) {
      [
        ['أبواب محتملة', data.door_count || 0, 'door'],
        ['شبابيك محتملة', data.window_count || 0, 'window'],
        ['فتحات للمراجعة', data.unclassified_opening_count || 0, 'opening'],
      ].forEach(([label, count, kind]) => {
        const badge = el('div', `opening-count ${kind}`);
        badge.append(el('span', '', label), el('strong', '', String(count)));
        openingSummary.append(badge);
      });

      const openingImage = document.getElementById('openings-overlay-image');
      openingImage.src = `${data.openings_overlay_url}?v=${Date.now()}`;
      document.getElementById('download-openings-overlay').href = openingImage.src;

      openings.forEach((opening) => {
        const card = el('article', `opening-card ${opening.type}`);
        const header = el('div', 'opening-card-head');
        header.append(
          el('h4', '', `${opening.id} · ${opening.type_label}`),
          el('strong', '', `${Math.round((opening.confidence || 0) * 100)}%`),
        );
        card.append(header);
        const size = opening.width_display || `${opening.width_pixels} px`;
        card.append(el('p', '', `العرض التقريبي ${size} · ${opening.orientation === 'horizontal' ? 'أفقي' : 'رأسي'}`));
        card.append(el('p', 'opening-reason', opening.evidence?.reason || 'لم تتوفر تفاصيل إضافية.'));
        openingList.append(card);
      });
    }
    results.hidden = false;
    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  async function runAnalysis(url, body) {
    errorBox.hidden = true;
    progressBox.hidden = false;
    analyzeButton.disabled = true;
    sampleButton.disabled = true;
    try {
      const response = await fetch(url, { method: 'POST', body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'فشل تحليل المخطط.');
      renderResult(data);
    } catch (error) {
      errorBox.textContent = error.message || 'تعذّر الاتصال بالخادم.';
      errorBox.hidden = false;
    } finally {
      progressBox.hidden = true;
      analyzeButton.disabled = !selectedFile;
      sampleButton.disabled = false;
    }
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!selectedFile) return;
    const body = new FormData();
    body.append('file', selectedFile);
    await runAnalysis('/api/analyze', body);
  });

  sampleButton.addEventListener('click', async () => {
    await runAnalysis('/api/sample', new FormData());
  });

  fetch('/health').then((response) => response.json()).then((data) => {
    const status = document.getElementById('health-status');
    status.classList.toggle('is-warning', !data.ocr_available);
    status.querySelector('span:last-child').textContent = data.ocr_available ? 'محرك القراءة جاهز' : 'التحليل الهندسي جاهز · OCR غير متاح';
  }).catch(() => {
    document.getElementById('health-status').querySelector('span:last-child').textContent = 'تعذّر فحص المحرك';
  });
})();

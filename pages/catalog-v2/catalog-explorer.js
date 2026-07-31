const FEED_URL = "./catalog-feed-v2.json";
const SOURCE_ORDER = ["scryfall", "tcgplayer"];
const SOURCE_LABELS = {
  scryfall: "Scryfall Catalogs",
  tcgplayer: "TCGplayer Catalogs",
};
const GAME_LABELS = {
  "magic-the-gathering": "Magic: The Gathering",
  yugioh: "Yu-Gi-Oh!",
  pokemon: "Pokémon",
  "flesh-and-blood": "Flesh and Blood",
  "digimon-card-game": "Digimon Card Game",
  "one-piece": "One Piece Card Game",
  lorcana: "Disney Lorcana",
  "star-wars-unlimited": "Star Wars: Unlimited",
  "pokemon-japan": "Pokémon Japan",
  "union-arena": "Union Arena",
  "gundam-card-game": "Gundam Card Game",
  riftbound: "Riftbound",
};

const number = new Intl.NumberFormat();
const dateTime = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "short",
});

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function formatDate(value) {
  return dateTime.format(new Date(value));
}

function relativeDate(value) {
  const seconds = Math.round((new Date(value).getTime() - Date.now()) / 1000);
  const formatter = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  const ranges = [
    [60, "second"],
    [60, "minute"],
    [24, "hour"],
    [7, "day"],
    [4.345, "week"],
    [12, "month"],
    [Infinity, "year"],
  ];
  let amount = seconds;
  for (const [limit, unit] of ranges) {
    if (Math.abs(amount) < limit) return formatter.format(Math.round(amount), unit);
    amount /= limit;
  }
  return formatter.format(Math.round(amount), "year");
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url} (${response.status})`);
  return response.json();
}

async function readJsonlGzip(url) {
  if (!("DecompressionStream" in globalThis)) {
    throw new Error("This browser cannot decompress catalog details.");
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url} (${response.status})`);
  const stream = response.body
    .pipeThrough(new DecompressionStream("gzip"))
    .pipeThrough(new TextDecoderStream());
  const reader = stream.getReader();
  const records = [];
  let pending = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    pending += value;
    const lines = pending.split("\n");
    pending = lines.pop() || "";
    for (const line of lines) {
      if (line) records.push(JSON.parse(line));
    }
  }
  if (pending) records.push(JSON.parse(pending));
  return records;
}

function loadHistory(entry) {
  const updates = Object.values(entry.updates).sort(
    (left, right) => left.to_version - right.to_version,
  );
  const stages = new Map();
  stages.set(entry.base.version, {
    version: entry.base.version,
    rows: entry.base.rows,
    source_updated_at: entry.base.source_updated_at,
    source: entry.descriptor.source,
    base: entry.base,
    update: entry.updates[String(entry.base.version)] || null,
  });
  let rows = entry.base.rows;
  for (const update of updates) {
    if (update.to_version < entry.base.version) continue;
    if (update.to_version > entry.base.version) {
      rows += update.rows.added - update.rows.deleted;
    }
    const stage = stages.get(update.to_version) || {
      version: update.to_version,
      rows,
      source_updated_at: update.source_updated_at,
      source: entry.descriptor.source,
      base: null,
      update: null,
    };
    stage.update = update;
    stage.rows = rows;
    stage.source_updated_at = update.source_updated_at;
    stages.set(update.to_version, stage);
  }
  return [...stages.values()].sort((left, right) => left.version - right.version);
}

function chip(text, kind = "") {
  return element("span", `chip ${kind}`.trim(), text);
}

function updateChips(stage) {
  if (!stage.update) return [chip(`${number.format(stage.rows)} base rows`)];
  const { rows, recognition_rows: recognitionRows, metadata_rows: metadataRows } = stage.update;
  const chips = [];
  if (rows.added) chips.push(chip(`${rows.added} added`, "added"));
  if (rows.updated) chips.push(chip(`${rows.updated} updated`, "updated"));
  if (rows.deleted) chips.push(chip(`${rows.deleted} deleted`, "deleted"));
  if (metadataRows) chips.push(chip(`${metadataRows} metadata rows`, "updated"));
  if (recognitionRows) chips.push(chip(`${recognitionRows} recognition rows`));
  return chips;
}

function renderUpdate(history, index) {
  const stage = history[index];
  const update = element("article", "update");
  const header = element("div", "update-header");
  const title = element("div");
  title.append(
    element(
      "h3",
      "",
      stage.version === 0 ? "Initial catalog" : `Version ${stage.version}`,
    ),
    element(
      "div",
      "update-meta",
      `${formatDate(stage.source_updated_at)} · ${number.format(stage.rows)} total rows`,
    ),
  );
  header.append(title);
  if (stage.update && index > 0) {
    const button = element("button", "", "Explore changes");
    button.type = "button";
    button.addEventListener("click", () => loadUpdateDetails(button, update, history, index));
    header.append(button);
  }
  const chips = element("div", "chips");
  chips.append(...updateChips(stage));
  update.append(header, chips);
  return update;
}

async function loadCatalogBody(body, entry) {
  if (body.dataset.loaded) return;
  body.dataset.loaded = "true";
  const loading = element("div", "loading", "Loading update history...");
  body.append(loading);
  try {
    const history = loadHistory(entry);
    loading.remove();
    const description = element(
      "p",
      "catalog-description",
      entry.descriptor.description,
    );
    const updates = element("div", "updates");
    for (let index = history.length - 1; index >= 0; index -= 1) {
      updates.append(renderUpdate(history, index));
    }
    body.append(description, updates);
  } catch (error) {
    loading.className = "error";
    loading.textContent = error.message;
  }
}

function renderCatalog(item) {
  const { key, entry, descriptor } = item;
  const details = element("details", "catalog");
  const summary = element("summary");
  const title = element("div", "catalog-title");
  title.append(
    element("strong", "", GAME_LABELS[descriptor.game] || descriptor.game),
    element("span", "", `${descriptor.profile} · ${descriptor.result_identifier}`),
  );
  const count = element("div", "catalog-count");
  count.append(
    element("strong", "", number.format(entry.rows)),
    element("span", "", "catalog rows"),
  );
  summary.append(
    title,
    element(
      "div",
      "catalog-freshness",
      `Source updated ${relativeDate(entry.source_updated_at)}`,
    ),
    count,
  );
  const body = element("div", "catalog-body");
  body.dataset.catalogKey = key;
  details.addEventListener("toggle", () => {
    if (details.open) loadCatalogBody(body, entry);
  });
  details.append(summary, body);
  return details;
}

function renderSections(items) {
  const container = document.querySelector("#catalog-sections");
  for (const source of SOURCE_ORDER) {
    const catalogs = items
      .filter((item) => item.descriptor.source === source)
      .sort((left, right) => right.entry.rows - left.entry.rows);
    if (!catalogs.length) continue;
    const section = element("section", "source-section");
    const heading = element("div", "source-heading");
    heading.append(
      element("h2", "", SOURCE_LABELS[source] || source),
      element("span", "", `${catalogs.length} supported ${catalogs.length === 1 ? "catalog" : "catalogs"}`),
    );
    section.append(heading, ...catalogs.map(renderCatalog));
    container.append(section);
  }
}

function loadCurrentDescriptors(feed) {
  return Object.entries(feed.families).flatMap(([family, familyEntry]) =>
    Object.entries(familyEntry.catalogs).map(([key, entry]) => ({
      key: `${family}/${key}`,
      entry,
      descriptor: entry.descriptor,
    })),
  );
}

function identityKey(value, source) {
  const record = value.record || value;
  const faceIndex = record.face_index || 0;
  return faceIndex ? `${source}:${record.id}:face:${faceIndex}` : `${source}:${record.id}`;
}

function publicIdentifiers(record, source) {
  if (!record) return {};
  const primaryNamespace = source === "scryfall" ? "scryfall_card" : "tcgplayer_product";
  return {
    [primaryNamespace]: record.id,
    ...record.identifiers,
  };
}

function loadBaseRecords(records, source) {
  const state = new Map();
  for (const line of records) {
    const { metadata, ...record } = line;
    state.set(identityKey(line, source), { record, metadata: metadata ?? null });
  }
  return state;
}

function applyRecordOperations(state, operations, source) {
  for (const operation of operations) {
    const key = identityKey(operation, source);
    if (operation.op === "delete") {
      state.delete(key);
      continue;
    }
    const previous = state.get(key);
    const metadata =
      "metadata" in operation ? operation.metadata : previous ? previous.metadata : null;
    state.set(key, { record: operation.record, metadata });
  }
}

async function reconstructPriorState(history, targetIndex) {
  let baseIndex = targetIndex - 1;
  while (baseIndex >= 0 && !history[baseIndex].base) baseIndex -= 1;
  if (baseIndex < 0) throw new Error("No usable base exists for this update.");
  const base = history[baseIndex];
  const records = await readJsonlGzip(base.base.assets.records.url);
  const state = loadBaseRecords(records, base.source);
  for (let index = baseIndex + 1; index < targetIndex; index += 1) {
    const stage = history[index];
    const operations = await readJsonlGzip(stage.update.assets.records.url);
    applyRecordOperations(state, operations, stage.source);
  }
  return state;
}

function changedFields(previous, current) {
  const fields = new Set([...Object.keys(previous || {}), ...Object.keys(current || {})]);
  return [...fields]
    .filter((field) => JSON.stringify(previous?.[field]) !== JSON.stringify(current?.[field]))
    .map((field) => ({
      field,
      previous: previous?.[field],
      current: current?.[field],
    }));
}

function setIdentity(metadata) {
  return metadata?.set || metadata?.set_name || "";
}

function describeRecord(key, recognition, metadata) {
  const promo =
    metadata?.promo === true ||
    metadata?.set_type === "promo" ||
    (Array.isArray(metadata?.promo_types) && metadata.promo_types.length > 0);
  return {
    key,
    name: recognition?.name || key,
    promo,
    context: [
      metadata?.set_name || metadata?.set,
      metadata?.collector_number && `#${metadata.collector_number}`,
      metadata?.lang && metadata.lang.toUpperCase(),
      promo && "Promo",
    ]
      .filter(Boolean)
      .join(" · "),
  };
}

function sourcePage(identifiers) {
  if (identifiers?.scryfall_card) {
    return {
      label: "View on Scryfall",
      url: `https://scryfall.com/card/${encodeURIComponent(identifiers.scryfall_card)}`,
    };
  }
  if (identifiers?.tcgplayer_product) {
    return {
      label: "View on TCGplayer",
      url: `https://www.tcgplayer.com/product/${encodeURIComponent(
        identifiers.tcgplayer_product,
      )}`,
    };
  }
  return null;
}

async function analyzeUpdate(history, index) {
  const prior = await reconstructPriorState(history, index);
  const stage = history[index];
  const operations = await readJsonlGzip(stage.update.assets.records.url);
  const changes = new Map();
  const targetMetadataChanges = new Map();
  for (const operation of operations) {
    const key = identityKey(operation, stage.source);
    const previousEntry = prior.get(key);
    if (operation.op === "delete") {
      changes.set(key, {
        ...describeRecord(key, previousEntry?.record, previousEntry?.metadata),
        kinds: ["deleted"],
        fields: [],
        identifiers: publicIdentifiers(previousEntry?.record, stage.source),
        metadata: previousEntry?.metadata,
        metadataLabel: "Previous metadata",
      });
      continue;
    }
    const existed = previousEntry !== undefined;
    const recognitionChanged = "embedding_index" in operation;
    const metadataChanged = "metadata" in operation;
    const currentMetadata = metadataChanged
      ? operation.metadata
      : existed
        ? previousEntry.metadata
        : null;
    const kinds = [];
    if (!existed) {
      kinds.push("new card");
    } else if (recognitionChanged) {
      const identical =
        JSON.stringify(previousEntry.record) === JSON.stringify(operation.record);
      kinds.push(identical ? "image updated" : "recognition updated");
    }
    let fields = [];
    if (metadataChanged) {
      targetMetadataChanges.set(key, currentMetadata);
      fields = changedFields(existed ? previousEntry.metadata : undefined, currentMetadata);
      if (currentMetadata === null) kinds.push("metadata removed");
      else if (existed && previousEntry.metadata) kinds.push("metadata corrected");
      else kinds.push("metadata added");
    }
    if (!kinds.length) kinds.push("updated");
    changes.set(key, {
      ...describeRecord(key, operation.record, currentMetadata),
      kinds,
      fields,
      identifiers: publicIdentifiers(operation.record, stage.source),
      metadata: currentMetadata,
      metadataLabel: "Metadata",
    });
  }
  const previousSets = new Set(
    [...prior.values()].map((entry) => setIdentity(entry.metadata)).filter(Boolean),
  );
  const newSets = new Map();
  for (const metadata of targetMetadataChanges.values()) {
    const identity = setIdentity(metadata);
    if (identity && !previousSets.has(identity)) {
      newSets.set(identity, metadata?.set_name || metadata?.set || identity);
    }
  }
  const rows = [...changes.values()];
  return {
    changes: rows,
    newSets: [...newSets.values()].sort(),
    counts: {
      newCards: rows.filter((change) => change.kinds.includes("new card")).length,
      newPromos: rows.filter(
        (change) => change.promo && change.kinds.includes("new card"),
      ).length,
      images: rows.filter((change) => change.kinds.includes("image updated")).length,
      recognition: rows.filter((change) => change.kinds.includes("recognition updated")).length,
      metadata: rows.filter((change) =>
        change.kinds.some((kind) => kind.startsWith("metadata")),
      ).length,
      deleted: rows.filter((change) => change.kinds.includes("deleted")).length,
    },
  };
}

function metric(value, label) {
  const node = element("div", "metric");
  node.append(element("strong", "", number.format(value)), element("span", "", label));
  return node;
}

function renderAnalysis(target, analysis) {
  const summary = element("div", "detail-summary");
  summary.append(
    metric(analysis.counts.newCards, "new cards"),
    metric(analysis.counts.newPromos, "new promos"),
    metric(analysis.counts.images, "updated images"),
    metric(analysis.counts.recognition, "recognition updates"),
    metric(analysis.counts.metadata, "metadata changes"),
    metric(analysis.counts.deleted, "deleted"),
  );
  target.append(summary);
  if (analysis.newSets.length) {
    const sets = element("section", "new-sets");
    const list = element("div", "set-list");
    list.append(...analysis.newSets.map((name) => element("span", "set-name", name)));
    sets.append(element("h4", "", "New sets"), list);
    target.append(sets);
  }
  const section = element("section", "change-list");
  section.append(element("h4", "", `Changed rows (${number.format(analysis.changes.length)})`));
  const pageSize = 100;
  let rendered = 0;
  const renderPage = () => {
    const page = analysis.changes.slice(rendered, rendered + pageSize);
    for (const change of page) {
      const row = element("details", "change-row");
      const rowSummary = element("summary", "change-summary");
      const description = element("div");
      description.append(
        element("div", "change-name", change.name),
        element("div", "change-context", change.context || change.key),
      );
      rowSummary.append(
        description,
        element("span", "change-kind", change.kinds.join(" + ")),
      );
      const expanded = element("div", "card-details");
      expanded.append(
        detailSection("Identification", {
          key: change.key,
          ...change.identifiers,
        }),
      );
      if (change.metadata) {
        expanded.append(detailSection(change.metadataLabel, change.metadata));
      }
      if (change.fields.length) {
        const changed = element("section", "detail-section");
        changed.append(element("h5", "", "Metadata changes"));
        const fields = element("div", "change-fields");
        for (const difference of change.fields) {
          fields.append(
            element(
              "div",
              "",
              `${difference.field}: ${displayValue(difference.previous)} → ${displayValue(difference.current)}`,
            ),
          );
        }
        changed.append(fields);
        expanded.append(changed);
      }
      const pageLink = sourcePage(change.identifiers);
      if (pageLink) {
        const link = element("a", "source-link", pageLink.label);
        link.href = pageLink.url;
        link.target = "_blank";
        link.rel = "noreferrer";
        expanded.append(link);
      }
      row.append(rowSummary, expanded);
      section.append(row);
    }

    function detailSection(title, values) {
      const section = element("section", "detail-section");
      section.append(element("h5", "", title));
      const list = element("dl", "detail-list");
      for (const [name, value] of Object.entries(values)) {
        const term = element("dt", "", name);
        const definition = element("dd", "", displayValue(value));
        list.append(term, definition);
      }
      section.append(list);
      return section;
    }

    function displayValue(value) {
      if (value === undefined) return "not set";
      if (typeof value === "string") return value;
      return JSON.stringify(value);
    }
    rendered += page.length;
    more.remove();
    if (rendered < analysis.changes.length) section.append(more);
  };
  const more = element("button", "", "Show more changes");
  more.type = "button";
  more.addEventListener("click", renderPage);
  renderPage();
  target.append(section);
}

async function loadUpdateDetails(button, update, history, index) {
  const existing = update.querySelector(".update-details");
  if (existing) {
    if (existing.dataset.failed === "true") {
      existing.remove();
    } else {
      existing.hidden = !existing.hidden;
      button.textContent = existing.hidden ? "Explore changes" : "Hide details";
      return;
    }
  }
  button.disabled = true;
  button.textContent = "Loading catalog data...";
  const details = element("div", "update-details");
  update.append(details);
  try {
    const analysis = await analyzeUpdate(history, index);
    renderAnalysis(details, analysis);
    button.textContent = "Hide details";
  } catch (error) {
    details.dataset.failed = "true";
    details.append(element("div", "error", error.message));
    button.textContent = "Try again";
  } finally {
    button.disabled = false;
  }
}

async function main() {
  try {
    const feed = await fetchJson(FEED_URL);
    document.querySelector("#catalog-status").textContent =
      `Last checked ${relativeDate(feed.checked_at)} · ${formatDate(feed.checked_at)}`;
    renderSections(loadCurrentDescriptors(feed));
  } catch (error) {
    const target = document.querySelector("#catalog-error");
    target.hidden = false;
    target.textContent = error.message;
    document.querySelector("#catalog-status").textContent = "Catalog status unavailable";
  }
}

export { analyzeUpdate, identityKey, loadHistory, reconstructPriorState };

if (typeof document !== "undefined") main();

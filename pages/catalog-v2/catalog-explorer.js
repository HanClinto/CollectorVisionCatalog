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
  if (!("DecompressionStream" in window)) {
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

function catalogRoot(entry) {
  return entry.base.manifest.url.replace(/\/version\/\d+\/manifest\.json$/, "");
}

function assetUrl(manifestUrl, asset) {
  return new URL(asset.path, manifestUrl).href;
}

async function loadHistory(entry) {
  const root = catalogRoot(entry);
  return Promise.all(
    Array.from({ length: entry.current_version + 1 }, (_, version) =>
      fetchJson(`${root}/version/${version}/manifest.json`).then((manifest) => ({
        manifest,
        url: `${root}/version/${version}/manifest.json`,
      })),
    ),
  );
}

function chip(text, kind = "") {
  return element("span", `chip ${kind}`.trim(), text);
}

function updateChips(manifest) {
  if (!manifest.delta) return [chip(`${number.format(manifest.rows)} initial rows`)];
  const { recognition, metadata } = manifest.delta;
  const chips = [];
  if (recognition.added) chips.push(chip(`${recognition.added} added`, "added"));
  if (recognition.updated) {
    chips.push(chip(`${recognition.updated} recognition updates`, "updated"));
  }
  if (recognition.deleted) chips.push(chip(`${recognition.deleted} deleted`, "deleted"));
  if (metadata.updated) chips.push(chip(`${metadata.updated} metadata fixes`, "updated"));
  return chips.length ? chips : [chip(`${manifest.delta.rows} rows changed`)];
}

function renderUpdate(history, index) {
  const { manifest } = history[index];
  const update = element("article", "update");
  const header = element("div", "update-header");
  const title = element("div");
  title.append(
    element("h3", "", index === 0 ? "Initial catalog" : `Version ${manifest.version}`),
    element(
      "div",
      "update-meta",
      `${formatDate(manifest.source_revision.updated_at)} · ${number.format(manifest.rows)} total rows`,
    ),
  );
  header.append(title);
  if (manifest.delta) {
    const button = element("button", "", "Explore changes");
    button.type = "button";
    button.addEventListener("click", () => loadUpdateDetails(button, update, history, index));
    header.append(button);
  }
  const chips = element("div", "chips");
  chips.append(...updateChips(manifest));
  update.append(header, chips);
  return update;
}

async function loadCatalogBody(body, entry) {
  if (body.dataset.loaded) return;
  body.dataset.loaded = "true";
  const loading = element("div", "loading", "Loading update history...");
  body.append(loading);
  try {
    const history = await loadHistory(entry);
    loading.remove();
    const description = element(
      "p",
      "catalog-description",
      history.at(-1).manifest.descriptor.description,
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

async function loadCurrentDescriptors(feed) {
  return Promise.all(
    Object.entries(feed.catalogs).map(async ([key, entry]) => {
      const manifest = await fetchJson(entry.base.manifest.url);
      return { key, entry, descriptor: manifest.descriptor };
    }),
  );
}

function applyIdentifierOperations(state, records) {
  for (const operation of records) {
    const key = operation.record?.key || operation.key;
    if (operation.op === "delete") state.delete(key);
    else state.set(key, operation.record);
  }
}

function applyMetadataOperations(state, records) {
  for (const operation of records) {
    if (operation.op === "delete") state.delete(operation.key);
    else state.set(operation.key, operation.metadata);
  }
}

async function reconstructPriorState(history, targetIndex) {
  let baseIndex = targetIndex - 1;
  while (baseIndex >= 0 && !history[baseIndex].manifest.base) baseIndex -= 1;
  if (baseIndex < 0) throw new Error("No usable base exists for this update.");
  const base = history[baseIndex];
  const [identifierRows, metadataRows] = await Promise.all([
    readJsonlGzip(assetUrl(base.url, base.manifest.base.identifiers)),
    readJsonlGzip(assetUrl(base.url, base.manifest.base.metadata)),
  ]);
  const identifiers = new Map(identifierRows.map((record) => [record.key, record]));
  const metadata = new Map(metadataRows.map((record) => [record.key, record.metadata]));
  for (let index = baseIndex + 1; index < targetIndex; index += 1) {
    const stage = history[index];
    const assets = stage.manifest.delta.assets;
    const jobs = [];
    if (assets.identifiers) {
      jobs.push(
        readJsonlGzip(assetUrl(stage.url, assets.identifiers)).then((records) =>
          applyIdentifierOperations(identifiers, records),
        ),
      );
    }
    if (assets.metadata) {
      jobs.push(
        readJsonlGzip(assetUrl(stage.url, assets.metadata)).then((records) =>
          applyMetadataOperations(metadata, records),
        ),
      );
    }
    await Promise.all(jobs);
  }
  return { identifiers, metadata };
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

function describeRecord(key, metadata) {
  const promo =
    metadata?.promo === true ||
    metadata?.set_type === "promo" ||
    (Array.isArray(metadata?.promo_types) && metadata.promo_types.length > 0);
  return {
    key,
    name: metadata?.name || key,
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

async function analyzeUpdate(history, index) {
  const prior = await reconstructPriorState(history, index);
  const stage = history[index];
  const assets = stage.manifest.delta.assets;
  const [identifierOps, metadataOps] = await Promise.all([
    assets.identifiers ? readJsonlGzip(assetUrl(stage.url, assets.identifiers)) : [],
    assets.metadata ? readJsonlGzip(assetUrl(stage.url, assets.metadata)) : [],
  ]);
  const targetMetadata = new Map(
    metadataOps
      .filter((operation) => operation.op !== "delete")
      .map((operation) => [operation.key, operation.metadata]),
  );
  const changes = new Map();
  for (const operation of identifierOps) {
    const key = operation.record?.key || operation.key;
    const previousRecord = prior.identifiers.get(key);
    const existed = previousRecord !== undefined;
    const metadata = targetMetadata.get(key) || prior.metadata.get(key);
    let kind;
    if (operation.op === "delete") kind = "deleted";
    else if (!existed) kind = "new card";
    else if (JSON.stringify(previousRecord) === JSON.stringify(operation.record)) {
      kind = "image updated";
    } else {
      kind = "recognition updated";
    }
    changes.set(key, {
      ...describeRecord(key, metadata),
      kinds: [kind],
      fields: [],
      imageUrl: operation.state?.image_url,
    });
  }
  for (const operation of metadataOps) {
    const previous = prior.metadata.get(operation.key);
    const current = operation.op === "delete" ? undefined : operation.metadata;
    const fields = changedFields(previous, current);
    const kind = operation.op === "delete"
      ? "metadata removed"
      : previous
        ? "metadata corrected"
        : "metadata added";
    const existing = changes.get(operation.key);
    if (existing) {
      if (!existing.kinds.includes("new card")) {
        existing.kinds.push(kind);
        existing.fields = fields;
      }
    } else {
      changes.set(operation.key, {
        ...describeRecord(operation.key, current || previous),
        kinds: [kind],
        fields,
      });
    }
  }
  const previousSets = new Set(
    [...prior.metadata.values()].map(setIdentity).filter(Boolean),
  );
  const newSets = new Map();
  for (const metadata of targetMetadata.values()) {
    const identity = setIdentity(metadata);
    if (identity && !previousSets.has(identity)) {
      newSets.set(identity, metadata.set_name || metadata.set || identity);
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
      const row = element("div", "change-row");
      const description = element("div");
      description.append(
        element("div", "change-name", change.name),
        element("div", "change-context", change.context || change.key),
      );
      if (change.fields.length) {
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
        description.append(fields);
      }
      if (change.imageUrl && safeHttpUrl(change.imageUrl)) {
        const imageLink = element("a", "image-link", "View current source image");
        imageLink.href = change.imageUrl;
        imageLink.target = "_blank";
        imageLink.rel = "noreferrer";
        description.append(imageLink);
      }
      row.append(description, element("span", "change-kind", change.kinds.join(" + ")));
      section.append(row);
    }

    function displayValue(value) {
      if (value === undefined) return "not set";
      if (typeof value === "string") return value;
      return JSON.stringify(value);
    }

    function safeHttpUrl(value) {
      try {
        return ["http:", "https:"].includes(new URL(value).protocol);
      } catch {
        return false;
      }
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
    renderSections(await loadCurrentDescriptors(feed));
  } catch (error) {
    const target = document.querySelector("#catalog-error");
    target.hidden = false;
    target.textContent = error.message;
    document.querySelector("#catalog-status").textContent = "Catalog status unavailable";
  }
}

main();

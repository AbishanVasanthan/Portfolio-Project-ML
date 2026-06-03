-- ============================================================
-- sku_schema.sql
-- Run once in Supabase SQL Editor after schema.sql
-- Creates tc_skus, tc_sku_demand_panel, tc_sku_forecasts
-- ============================================================

-- ── 1. Product catalogue ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS tc_skus (
    sku_id     SERIAL PRIMARY KEY,
    sku_code   VARCHAR(20) UNIQUE NOT NULL,
    name       VARCHAR(100) NOT NULL,
    category   VARCHAR(50) DEFAULT 'specialty',
    mix_ratio  DECIMAL(5,4) NOT NULL,   -- baseline share of total depot demand
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 6 Tokyo Cement specialty products
-- mix_ratio sums to 1.00 (35+25+15+12+8+5)
INSERT INTO tc_skus (sku_code, name, mix_ratio) VALUES
    ('SUPERMIX',    'Tokyo SuperMix',    0.35),
    ('SUPERFIX',    'Tokyo SuperFix',    0.25),
    ('SUPERSEAL',   'Tokyo SuperSeal',   0.15),
    ('SUPERSET',    'Tokyo SuperSet',    0.12),
    ('SUPERSCREED', 'Tokyo SuperScreed', 0.08),
    ('SUPERFLOW',   'Tokyo SuperFlow',   0.05)
ON CONFLICT (sku_code) DO NOTHING;


-- ── 2. Per-product weekly demand panel ───────────────────────
-- Lean table: only SKU-specific demand columns.
-- Weather, economics, calendar are joined from tc_demand_panel
-- during training (no duplication).
CREATE TABLE IF NOT EXISTS tc_sku_demand_panel (
    id             BIGSERIAL PRIMARY KEY,
    depot_id       INTEGER NOT NULL REFERENCES tc_depots(depot_id),
    sku_id         INTEGER NOT NULL REFERENCES tc_skus(sku_id),
    week_start     DATE NOT NULL,
    demand_tonnes  DECIMAL(10, 2),
    sales_tonnes   DECIMAL(10, 2),
    data_source    VARCHAR(20) DEFAULT 'augmented',
    created_at     TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT tc_sku_demand_panel_unique UNIQUE (depot_id, sku_id, week_start)
);

CREATE INDEX IF NOT EXISTS idx_sku_panel_depot_week
    ON tc_sku_demand_panel (depot_id, week_start);
CREATE INDEX IF NOT EXISTS idx_sku_panel_sku_week
    ON tc_sku_demand_panel (sku_id, week_start);


-- ── 3. Per-product forecast store ────────────────────────────
CREATE TABLE IF NOT EXISTS tc_sku_forecasts (
    id               BIGSERIAL PRIMARY KEY,
    depot_id         INTEGER NOT NULL REFERENCES tc_depots(depot_id),
    sku_id           INTEGER NOT NULL REFERENCES tc_skus(sku_id),
    as_of_date       DATE NOT NULL,
    horizon_weeks    INTEGER NOT NULL,
    forecast_week    DATE NOT NULL,
    demand_forecast  DECIMAL(10, 2),
    model_version    VARCHAR(50),
    generated_at     TIMESTAMPTZ DEFAULT NOW(),
    -- one forecast per depot × SKU × target week (overwrite on re-run)
    CONSTRAINT tc_sku_forecasts_unique UNIQUE (depot_id, sku_id, forecast_week)
);

CREATE INDEX IF NOT EXISTS idx_sku_forecasts_depot_sku
    ON tc_sku_forecasts (depot_id, sku_id, as_of_date);

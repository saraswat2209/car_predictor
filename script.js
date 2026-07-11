const API_BASE = "";

let OPTIONS = null;

function fillSelect(el, values, selected) {
  el.innerHTML = "";
  values.forEach(v => {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    if (v === selected) opt.selected = true;
    el.appendChild(opt);
  });
}

async function loadOptions() {
  const res = await fetch(`${API_BASE}/api/options`);
  const data = await res.json();
  OPTIONS = data;

  fillSelect(document.getElementById("brand"), data.brands, data.brands[0]);
  updateModelsForBrand(data.brands[0]);
  fillSelect(document.getElementById("fuel_type"), data.fuel_types);
  fillSelect(document.getElementById("transmission"), data.transmissions);
  fillSelect(document.getElementById("owner_type"), data.owner_types);
  fillSelect(document.getElementById("location"), data.locations);

  document.getElementById("modelChipText").textContent =
    `${data.model_name} · R² ${data.r2_score}`;
}

function updateModelsForBrand(brand) {
  const models = OPTIONS.models_by_brand[brand] || [];
  fillSelect(document.getElementById("model"), models, models[0]);
}

document.addEventListener("DOMContentLoaded", () => {
  loadOptions();

  document.getElementById("brand").addEventListener("change", (e) => {
    updateModelsForBrand(e.target.value);
  });

  document.getElementById("predictForm").addEventListener("submit", async (e) => {
    e.preventDefault();

    const payload = {
      brand: document.getElementById("brand").value,
      model: document.getElementById("model").value,
      year: document.getElementById("year").value,
      km_driven: document.getElementById("km_driven").value,
      owner_type: document.getElementById("owner_type").value,
      fuel_type: document.getElementById("fuel_type").value,
      transmission: document.getElementById("transmission").value,
      location: document.getElementById("location").value,
      engine_cc: document.getElementById("engine_cc").value,
      power_bhp: document.getElementById("power_bhp").value,
      mileage_kmpl: document.getElementById("mileage_kmpl").value,
      seats: document.getElementById("seats").value,
    };

    const idle = document.getElementById("resultIdle");
    const live = document.getElementById("resultLive");
    const errorBox = document.getElementById("resultError");

    idle.classList.add("hidden");
    errorBox.classList.add("hidden");

    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      if (!data.success) throw new Error(data.error || "Prediction failed");

      document.getElementById("priceLakhs").textContent =
        `₹ ${data.predicted_price_lakhs.toFixed(2)} L`;
      document.getElementById("priceInr").textContent =
        `≈ ₹${data.predicted_price_inr.toLocaleString("en-IN")}`;
      document.getElementById("modelUsedText").textContent = data.model_used;

      const age = OPTIONS.current_year - parseInt(payload.year);
      document.getElementById("carAgeText").textContent = `${age} yr${age === 1 ? "" : "s"}`;

      const kmPerYear = Math.round(payload.km_driven / Math.max(age, 1));
      document.getElementById("usageText").textContent = `${kmPerYear.toLocaleString("en-IN")} km/yr`;

      live.classList.remove("hidden");
    } catch (err) {
      document.getElementById("errorText").textContent = err.message;
      errorBox.classList.remove("hidden");
    }
  });
});

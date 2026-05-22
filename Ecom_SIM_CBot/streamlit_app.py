import streamlit as st
import numpy as np
# from transformers import AutoTokenizer, AutoModelForCausalLM (moved to lazy load)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Simulador de E-commerce",
    page_icon="🛒",
    layout="wide",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; }
[data-testid="stMetricDelta"] { font-size: 0.85rem; }

.story-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-left: 4px solid #e94560;
    border-radius: 10px;
    padding: 1.2rem 1.6rem;
    color: #eee;
    margin: 0.5rem 0 1.2rem 0;
    line-height: 1.7;
}

.tier-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
}
.tier-low    { background: #fde8e8; color: #c0392b; }
.tier-mid    { background: #fef9e7; color: #d68910; }
.tier-high   { background: #e8f8f5; color: #1a8a5a; }
.tier-prem   { background: #eaf0fb; color: #1a5276; }

.order-slider-wrapper {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 14px;
    padding: 1.4rem 1.8rem 0.8rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}
.order-slider-title {
    color: #f0c040;
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.order-slider-sub {
    color: #aac; font-size: 0.78rem; margin-bottom: 0.6rem;
}

.section-divider {
    border: none;
    border-top: 1px solid #e0e0e0;
    margin: 1.2rem 0;
}

.kpi-note { font-size: 0.75rem; color: #999; }
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
params  = np.load('model_params.npz', allow_pickle=True)
weights = np.array(params['weights']).flatten()
bias    = float(params['bias'].item())

# ── Feature metadata ───────────────────────────────────────────────────────────
FEATURES = [
    {
        "key": "time_on_site",
        "label": "⏱️ Tempo no Site (min)",
        "desc": "Quanto tempo o visitante passou navegando. Sessões mais longas indicam maior intenção de compra.",
        "min": 0.5, "max": 60.0, "default": 8.0, "step": 0.5, "format": "%.1f min",
    },
    {
        "key": "pages_visited",
        "label": "📄 Páginas Visitadas",
        "desc": "Número de páginas acessadas na sessão. Mais páginas = mais itens no carrinho.",
        "min": 1.0, "max": 30.0, "default": 5.0, "step": 1.0, "format": "%.0f páginas",
    },
    {
        "key": "items_in_cart",
        "label": "🛒 Itens no Carrinho",
        "desc": "Quantidade de produtos adicionados. Principal driver do valor do pedido.",
        "min": 1.0, "max": 20.0, "default": 3.0, "step": 1.0, "format": "%.0f itens",
    },
    {
        "key": "discount_pct",
        "label": "🏷️ Desconto Aplicado (%)",
        "desc": "Percentual de desconto concedido. Aumenta conversão, mas reduz valor unitário.",
        "min": 0.0, "max": 30.0, "default": 5.0, "step": 5.0, "format": "%.0f%%",
    },
    {
        "key": "is_returning",
        "label": "🔁 Cliente Recorrente?",
        "desc": "1 = já comprou antes; 0 = novo visitante. Recorrentes gastam mais.",
        "min": 0.0, "max": 1.0, "default": 0.0, "step": 1.0, "format": "%.0f (0=Novo / 1=Recorrente)",
    },
    {
        "key": "device_mobile",
        "label": "📱 Acesso por Celular?",
        "desc": "1 = smartphone; 0 = desktop. Mobile converte menos e com ticket levemente inferior.",
        "min": 0.0, "max": 1.0, "default": 1.0, "step": 1.0, "format": "%.0f (0=Desktop / 1=Mobile)",
    },
    {
        "key": "hour_of_day",
        "label": "🕐 Hora do Dia",
        "desc": "Hora da sessão (0–23h). Picos entre 18h–22h.",
        "min": 0.0, "max": 23.0, "default": 19.0, "step": 1.0, "format": "%.0fh",
    },
    {
        "key": "avg_item_price",
        "label": "💲 Preço Médio por Item (R$)",
        "desc": "Valor médio dos produtos no carrinho. Maior impacto individual no pedido.",
        "min": 10.0, "max": 500.0, "default": 150.0, "step": 10.0, "format": "R$ %.0f",
    },
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def order_story(order_val, items, discount, returning):
    if order_val < 80:
        tier, emoji, badge = "Ticket Baixo", "📦", "tier-low"
        msg = (f"R$ {order_val:.2f} — pedido de baixo ticket. "
               "Considere upsell como 'compre mais e ganhe frete grátis'.")
    elif order_val < 250:
        tier, emoji, badge = "Ticket Médio", "🛍️", "tier-mid"
        msg = (f"R$ {order_val:.2f} — ticket dentro da média. "
               "Ative recomendações de produtos complementares (cross-sell).")
    elif order_val < 600:
        tier, emoji, badge = "Ticket Alto", "💼", "tier-high"
        msg = (f"R$ {order_val:.2f} — ticket elevado! "
               "Ative programas de fidelidade e cashback.")
    else:
        tier, emoji, badge = "Ticket Premium", "💎", "tier-prem"
        msg = (f"R$ {order_val:.2f} — compra premium. "
               "Priorize atendimento VIP e pós-venda proativo.")
    if returning == 1.0:
        msg += " ♻️ **Cliente recorrente** — segmente em campanhas de retenção."
    if discount >= 20:
        msg += " ⚠️ Desconto alto aplicado — monitore a margem."
    return emoji, tier, badge, msg


def estimate_kpis(order_val, items, discount, returning, mobile):
    aov       = order_val
    conv_rate = 0.025 + returning * 0.04 - mobile * 0.008 + (discount / 100) * 0.03
    conv_rate = min(max(conv_rate, 0.005), 0.12) * 100
    monthly_sessions = 1000
    monthly_revenue  = monthly_sessions * (conv_rate / 100) * aov
    abandonment = min(max(100 - conv_rate * 5, 55), 85)
    rpv         = monthly_revenue / monthly_sessions
    return aov, conv_rate, monthly_revenue, abandonment, rpv


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🛒 Configure a Sessão")
st.sidebar.markdown(
    "Simule o perfil de um visitante e veja as métricas estimadas pelo modelo. "
    "Cada parâmetro impacta o valor do pedido e os KPIs derivados."
)
st.sidebar.markdown("---")

feature_values = []
for feat in FEATURES:
    st.sidebar.markdown(f"**{feat['label']}**")
    st.sidebar.caption(feat["desc"])
    v = st.sidebar.slider(
        label=feat["label"],
        min_value=float(feat["min"]),
        max_value=float(feat["max"]),
        value=float(feat["default"]),
        step=float(feat["step"]),
        label_visibility="collapsed",
        key=feat["key"],
    )
    feature_values.append(v)
    st.sidebar.markdown("")

# ── Model prediction ───────────────────────────────────────────────────────────
X             = np.array(feature_values).reshape(1, -1)
model_pred    = float(np.dot(X, weights).item() + bias)
model_pred    = max(model_pred, 10.0)

fv = {f["key"]: v for f, v in zip(FEATURES, feature_values)}

# ── Main page ──────────────────────────────────────────────────────────────────
st.title("🛒 Simulador de Métricas de E-commerce")
st.markdown(
    "Modelo de **Regressão Linear** treinado em 5 000 sessões sintéticas. "
    "Ajuste os parâmetros na sidebar **ou** use o slider abaixo para "
    "sobrescrever o valor do pedido manualmente."
)
st.markdown("---")

# ── 💰 Order Value Slider (main page) ─────────────────────────────────────────
st.markdown('<div class="order-slider-wrapper">', unsafe_allow_html=True)
st.markdown('<div class="order-slider-title">💰 Valor do Pedido (R$)</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="order-slider-sub">Predição do modelo: <b>R$ {model_pred:,.2f}</b> — '
    f'arraste para simular cenários manuais</div>',
    unsafe_allow_html=True,
)
order_val = st.slider(
    label="Valor do Pedido",
    min_value=10.0,
    max_value=2000.0,
    value=round(model_pred, 1),
    step=5.0,
    format="R$ %.0f",
    label_visibility="collapsed",
    key="order_val_override",
)
delta_vs_model = order_val - model_pred
delta_color = "#27ae60" if delta_vs_model >= 0 else "#e74c3c"
delta_sign  = "+" if delta_vs_model >= 0 else ""
st.markdown(
    f'<span class="kpi-note">Δ vs modelo: <b style="color:{delta_color}">R$ {delta_sign}{delta_vs_model:,.2f}</b></span>',
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)

# ── Derived values ─────────────────────────────────────────────────────────────
emoji, tier, badge, story = order_story(order_val, fv["items_in_cart"], fv["discount_pct"], fv["is_returning"])
aov, conv_rate, monthly_rev, abandonment, rpv = estimate_kpis(
    order_val, fv["items_in_cart"], fv["discount_pct"], fv["is_returning"], fv["device_mobile"]
)

# Reference KPIs at model prediction (for deltas)
_, conv_ref, rev_ref, aband_ref, rpv_ref = estimate_kpis(
    model_pred, fv["items_in_cart"], fv["discount_pct"], fv["is_returning"], fv["device_mobile"]
)

# ── KPI row ────────────────────────────────────────────────────────────────────
st.subheader("📊 KPIs em Tempo Real")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(
    "💰 Valor do Pedido",
    f"R$ {order_val:,.2f}",
    delta=f"R$ {delta_vs_model:+,.2f} vs modelo",
    delta_color="normal",
)
c2.metric(
    "📊 Categoria",
    f"{emoji} {tier}",
)
c3.metric(
    "🎯 Conversão",
    f"{conv_rate:.2f}%",
    delta=f"{conv_rate - conv_ref:+.2f}%",
    delta_color="normal",
)
c4.metric(
    "🚪 Abandono",
    f"{abandonment:.1f}%",
    delta=f"{abandonment - aband_ref:+.1f}%",
    delta_color="inverse",
)
c5.metric(
    "💵 Receita/Visitante",
    f"R$ {rpv:.2f}",
    delta=f"R$ {rpv - rpv_ref:+.2f}",
    delta_color="normal",
)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Monthly revenue ────────────────────────────────────────────────────────────
st.subheader("📈 Estimativa Mensal (1 000 sessões)")
m1, m2, m3 = st.columns(3)
m1.metric(
    "🏪 Receita Estimada/mês",
    f"R$ {monthly_rev:,.2f}",
    delta=f"R$ {monthly_rev - rev_ref:+,.2f} vs modelo",
    delta_color="normal",
)
m2.metric(
    "🛍️ Pedidos Convertidos",
    f"{monthly_rev / aov:.0f} pedidos",
)
m3.metric(
    "📦 Itens Vendidos/mês",
    f"{(monthly_rev / aov) * fv['items_in_cart']:.0f} itens",
)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Storytelling ───────────────────────────────────────────────────────────────
st.subheader(f"{emoji} O que esse pedido diz sobre este cliente?")
st.markdown(
    f'<span class="tier-badge {badge}">{emoji} {tier}</span>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="story-box">{story}</div>',
    unsafe_allow_html=True,
)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Feature impact table ───────────────────────────────────────────────────────
st.subheader("🔍 Impacto de cada variável no valor do pedido")
st.markdown(
    "Pesos **positivos** aumentam o ticket; pesos **negativos** reduzem. "
    "A contribuição é o produto do valor escolhido pelo peso do modelo."
)

impact = {
    "Variável":          [f["label"] for f in FEATURES],
    "Valor":             [f"{v:.2f}" for v in feature_values],
    "Peso":              [f"{w:+.4f}" for w in weights],
    "Contribuição (R$)": [f"{v * w:+.2f}" for v, w in zip(feature_values, weights)],
}
st.table(impact)

# ── Benchmarks ────────────────────────────────────────────────────────────────
st.subheader("📊 Benchmarks de E-commerce (referência de mercado)")
bench_data = {
    "Métrica":        ["Taxa de Conversão", "Abandono de Carrinho", "Ticket Médio", "Receita/Visitante"],
    "Setor (média)":  ["2–3%", "70–80%", "R$ 150–300", "R$ 3–6"],
    "Sua simulação":  [
        f"{conv_rate:.2f}%",
        f"{abandonment:.1f}%",
        f"R$ {aov:,.2f}",
        f"R$ {rpv:.2f}",
    ],
}
st.table(bench_data)

st.caption(
    "Modelo treinado com dataset sintético de 5 000 sessões. "
    "Métricas derivadas são estimativas baseadas em heurísticas de mercado."
)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Chatbot Insights ───────────────────────────────────────────────────────────
st.subheader("💬 Chat de Insights da Simulação")
st.markdown("Interaja com a IA para obter insights adicionais sobre os resultados da simulação.")

@st.cache_resource(show_spinner="Carregando Inteligência Artificial (Qwen2.5-0.5B-Instruct)...")
def get_llm():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(">>> Iniciando carregamento do modelo Qwen2.5-0.5B (FORCED CPU)...")
    try:
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=None,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32
        )
        model.to("cpu")
        print(">>> Modelo Qwen2.5-0.5B carregado com sucesso na CPU.")
        return tokenizer, model
    except Exception as e:
        print(f">>> AVISO: Falha ao carregar modelo real. Usando modo de fallback. Erro: {e}")
        return None, None
    except Exception as e:
        print(f">>> ERRO AO CARREGAR MODELO: {e}")
        raise e

try:
    tokenizer, model = get_llm()
    
    # Histórico de chat na sessão do Streamlit
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    user_query = st.chat_input("Pergunte algo sobre as variáveis e resultados...")

    if user_query:
        st.chat_message("user").write(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Format context from simulation
        context = (
            f"Variáveis selecionadas na simulação:\n"
            f"- Tempo no site: {fv['time_on_site']} min\n"
            f"- Páginas visitadas: {fv['pages_visited']}\n"
            f"- Itens no carrinho: {fv['items_in_cart']}\n"
            f"- Desconto: {fv['discount_pct']}%\n"
            f"- Cliente recorrente: {'Sim' if fv['is_returning']==1.0 else 'Não'}\n"
            f"- Acesso Mobile: {'Sim' if fv['device_mobile']==1.0 else 'Não'}\n"
            f"- Preço Médio por item: R${fv['avg_item_price']}\n\n"
            f"Métricas KPI simuladas:\n"
            f"- Valor do Pedido (AOV): R${order_val:.2f}\n"
            f"- Taxa de Conversão: {conv_rate:.2f}%\n"
            f"- Abandono estimado: {abandonment:.1f}%\n"
            f"- Receita mensal estimada: R${monthly_rev:.2f}\n"
        )
        
        prompt = f"Baseado nos seguintes dados do e-commerce:\n{context}\n\nResponda diretamente e de forma concisa e clara à seguinte pergunta do usuário: {user_query}"
        
        messages = [{"role": "user", "content": prompt}]
        
        with st.spinner("Analisando dados..."):
            try:
                if model is not None and tokenizer is not None:
                    inputs = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt"
                    ).to("cpu")
                    
                    import torch
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs, 
                            max_new_tokens=150,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    input_length = inputs["input_ids"].shape[-1]
                    response_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
                else:
                    raise Exception("Model not loaded")
                    
            except Exception as e:
                print(f">>> Fallback acionado: {e}")
                # Heuristic Fallback
                if "conversão" in user_query.lower() or "vender" in user_query.lower():
                    response_text = f"Para aumentar sua conversão de {conv_rate:.2f}%, foque em reduzir o abandono de {abandonment:.1f}% oferecendo frete grátis ou cupons de saída."
                elif "ticket" in user_query.lower() or "valor" in user_query.lower():
                    response_text = f"Seu ticket médio é R$ {order_val:.2f}. Tente estratégias de 'Compre Junto' para aumentar o número de itens no carrinho (atualmente {fv['items_in_cart']})."
                else:
                    response_text = f"Análise rápida: Com {fv['time_on_site']} min no site e {fv['pages_visited']} páginas, o interesse é real. Foque em personalização para converter este perfil."
            
            st.chat_message("assistant").write(response_text)
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})

except Exception as e:
    st.error(f"Erro ao carregar ou executar o modelo: {e}")
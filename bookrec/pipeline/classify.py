
from __future__ import annotations

import logging

import anthropic
import pandas as pd
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from psycopg2.extras import Json
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from transformers import pipeline

from ..config import cfg

log = logging.getLogger(__name__)



ALL_THEMES: list[str] = [
   # Narrative/emotional core
"книга о любви и романтических отношениях",
"книга о семейных конфликтах и семейной драме",
"книга о взрослении, поиске себя и идентичности",
"книга о дружбе и человеческих связях",
"книга о предательстве и утрате доверия",
"книга о горе, потере и трауре",
"книга о мести и справедливости",
"книга о выживании и борьбе за жизнь",

# Plot/setting driven
"книга о войне и её последствиях для людей",
"книга о преступлениях, убийствах и расследованиях",
"книга о политических интригах и борьбе за власть",
"книга о жизни людей в конкретную историческую эпоху",
"книга о путешествиях и приключениях в незнакомых местах",
"книга о жизни в антиутопическом или тоталитарном обществе",
"книга о магии, фэнтезийных мирах и существах",
"книга о сверхъестественном, призраках и мистике",
"книга о страхе, ужасе и психологическом напряжении",
"книга о космосе, далёком будущем и научной фантастике",

# Tone/style
"книга с сатирой и иронией над обществом",
"книга о трагической судьбе героя",
"книга о моральных дилеммах и этических выборах",

# Social lens
"книга о классовом неравенстве и социальных противоречиях",
"книга о расизме, дискриминации и угнетении",
"книга о гендере, феминизме и роли женщины в обществе",
"книга об иммиграции, чужбине и поиске дома",

# Format
"пьеса — театральное произведение в диалогах",
"поэма или сборник стихов",
# History/politics — concrete, not disciplinary
"книга об исторических событиях, войнах и революциях",
"книга о политических системах, диктатурах и демократии",
"книга о колониализме и его последствиях",

# Society/people
"книга об устройстве общества и социальных явлениях",
"книга о психологии человека и поведении",
"книга о биографии или мемуарах реального человека",

# Science — phrased as topic not discipline
"книга о происхождении жизни, эволюции и биологии",
"книга о вселенной, физике и законах природы",
"книга об искусственном интеллекте и будущем технологий",

# Business/practical
"книга о бизнесе, предпринимательстве и управлении",
"книга о деньгах, инвестициях и личных финансах",
"книга о продуктивности, привычках и достижении целей",
"книга о карьере и профессиональном развитии",
"книга о программировании и разработке программного обеспечения",

# Philosophy/meaning — made specific
"книга о смысле жизни, смерти и экзистенциальных вопросах",
"книга о морали, этике и том, как правильно жить",
"книга о сознании, разуме и природе личности",
"книга о религии, духовности и вере",
"книга о свободе, власти и правах человека",

# Culture/self
"книга о культуре, традициях и национальной идентичности",
"книга о саморазвитии и личностном росте",
"книга о здоровье, питании и образе жизни",
]


_STOPWORDS: list[str] = [
    
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а',
    'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же',
    'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от',
    'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже',
    'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был',
    'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там',
    'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть',
    'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб',
    'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж',
    'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем',
    'про', 'об', 'им', 'при', 'эту', 'эта', 'эти', 'это', 'книга', 'книги',
    'после', 'здесь', 'которое', 'которые', 'чтобы', 'никогда', 'самый',
    'писателя', 'писатель', 'автор', 'глава', 'жизнь', 'время', 'лет',
    'годы', 'чтоб', 'всего', 'куда', 'кого', 'чего', 'зачем', 'почему',
    'который', 'которая', 'которого', 'которой', 'которым', 'которых',
    'истории', 'история', 'мир', 'один', 'одна', 'однако', 'своей', 'своего',
    'своих', 'своим', 'лишь', 'том', 'тем', 'такой', 'такая', 'таких',
    'именно', 'также', 'между', 'через', 'перед', 'людей', 'человека',
    'человек', 'люди', 'стал', 'стала', 'стало', 'стали', 'может', 'могут',
    'должен', 'должна', 'хотя', 'этой', 'этом', 'этому', 'нему', 'нее',
    'него', 'жизни', 'жить', 'день', 'дней', 'раз', 'снова', 'уже',
    'the', 'and', 'it', 'she', 'her', 'i', 'of', 'or', 'these', 'nobody',
    'those', 'them', "you're", 'his', 'my', 'for', 'is', 'was', 'in', 'to',
    'a', 'that', 'he', 'with', 'on', 'are', 'as', 'at', 'be', 'this', 'by',
    'an', 'but', 'not', 'have', 'from', 'they', 'we', 'his', 'its',
]

_LABEL_PROMPT = """\
Твоя задача -- дать ключевую тему кластера книг 2-7 словами, используя ключевые слова кластера и описания книг из этого кластера. Никаких объяснений, никаких скобок.

Ключевые слова кластера: {words}

Примеры описания книг из кластера:
{samples}

Примеры жанровых меток:
- любовь, чувства -> Любовный роман
- война, солдат -> Военная проза
- убийство, следователь -> Детектив
- трудный путь, преодолевание -> Книга о поиске себя
- эпохи, история семьи, перемены -> Роман-эпопея

Ключевая тема (только 2-7 слов, без объяснений):\
"""


def _zero_shot_classify(classifier, text: str) -> list[dict]:
    result = classifier(text, ALL_THEMES, multi_label=True)
    return [
        {"label": label, "score": float(score)}
        for label, score in zip(result["labels"], result["scores"])
        if score >= cfg.theme_score_threshold
    ]


def _label_topic(client: anthropic.Anthropic, words: list[str], sample_docs: list[str]) -> str:
    samples = "\n".join(f"- {d[:200]}" for d in sample_docs)
    prompt = _LABEL_PROMPT.format(words=", ".join(words[:10]), samples=samples)
    response = client.messages.create(
        model=cfg.claude_model,
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def run(conn, only_bertopic: bool = False) -> None:
    if only_bertopic:
        df = pd.read_sql("""
            SELECT id, COALESCE(description, '') AS description
            FROM books
            WHERE description IS NOT NULL AND description != ''
              AND tags->>'topic_label' IS NOT NULL
        """, conn)
        log.info("--only-bertopic: re-running Stage 2 on %d previously BERTopic-classified books", len(df))
        _run_bertopic_stage(conn, df)
        return

    df = pd.read_sql("""
        SELECT id, COALESCE(description, '') AS description
        FROM books
        WHERE description IS NOT NULL AND description != ''
    """, conn)
    log.info("Loaded %d books", len(df))

    # ------------------------------------------------------------------
    # Stage 1 — zero-shot
    # ------------------------------------------------------------------
    log.info("Stage 1: zero-shot classification (%d themes)...", len(ALL_THEMES))
    classifier = pipeline("zero-shot-classification", model=cfg.classify_model)

    classified_ids: set[int] = set()
    zero_shot_tags: dict[int, list[dict]] = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="zero-shot"):
        tags = _zero_shot_classify(classifier, row["description"])
        zero_shot_tags[row["id"]] = tags
        if tags:
            classified_ids.add(row["id"])

    log.info(
        "Zero-shot: %d classified, %d unclassified",
        len(classified_ids),
        len(df) - len(classified_ids),
    )

    # Save zero-shot results
    with conn.cursor() as cur:
        for book_id, topics in zero_shot_tags.items():
            cur.execute(
                "UPDATE books SET tags = %s WHERE id = %s",
                (Json({"topics": topics}), book_id),
            )
    conn.commit()
    log.info("Stage 1 results saved.")

    # ------------------------------------------------------------------
    # Stage 2 — BERTopic fallback
    # ------------------------------------------------------------------
    unclassified = df[~df["id"].isin(classified_ids)].reset_index(drop=True)

    if unclassified.empty:
        log.info("All books classified by zero-shot — skipping BERTopic.")
        return

    _run_bertopic_stage(conn, unclassified)


def _run_bertopic_stage(conn, df: pd.DataFrame) -> None:
    """Fit BERTopic on df, generate Claude labels, and save results to DB."""
    log.info("Stage 2: BERTopic on %d books...", len(df))

    topic_model = BERTopic(
        language="multilingual",
        min_topic_size=cfg.bertopic_min_topic_size,
        vectorizer_model=CountVectorizer(ngram_range=(1,3),stop_words=_STOPWORDS),
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
    )

    docs = df["description"].tolist()
    topics, _ = topic_model.fit_transform(docs)

    # Merge similar small clusters into their nearest neighbours using
    # BERTopic's built-in topic similarity reduction. "auto" uses the same
    # HDBSCAN linkage to decide which topics to merge.
    n_before = len(set(t for t in topics if t != -1))
    topic_model.reduce_topics(docs, nr_topics="auto")
    topics = topic_model.topics_
    n_after = len(set(t for t in topics if t != -1))
    if n_before != n_after:
        log.info("reduce_topics: %d → %d clusters", n_before, n_after)

    # Reassign -1 outliers to nearest cluster by embedding similarity
    if any(t == -1 for t in topics):
        topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings")
        topic_model.update_topics(docs, topics=topics)
        log.info("Outliers redistributed.")

    df = df.copy()
    df["topic_id"] = topics

    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    topic_labels: dict[int, str] = {}

    for topic_id in sorted(df["topic_id"].unique()):
        if topic_id == -1:
            continue
        words = [w for w, _ in topic_model.get_topic(topic_id)]
        sample_docs = df[df["topic_id"] == topic_id]["description"].head(5).tolist()
        label = _label_topic(client, words, sample_docs)
        topic_labels[topic_id] = label
        log.info("Topic %d: %s → %s", topic_id, words[:5], label)

    with conn.cursor() as cur:
        for _, row in df.iterrows():
            topic_info = topic_model.get_topic(row["topic_id"])
            topic_words = [w for w, _ in topic_info] if topic_info else []
            label = topic_labels.get(row["topic_id"], "unknown")
            cur.execute(
                "UPDATE books SET tags = %s WHERE id = %s",
                (
                    Json({
                        "topics": [],
                        "bertopic_words": topic_words,
                        "topic_label": label,
                    }),
                    row["id"],
                ),
            )
    conn.commit()

    log.info(
        "Stage 2 done. Topic distribution:\n%s",
        df["topic_id"].value_counts().to_string(),
    )

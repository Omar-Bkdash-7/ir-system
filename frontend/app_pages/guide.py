"""
User Guide Page - Bilingual (EN/AR)
"""
import streamlit as st


def render():
    st.markdown("## 📚 User Guide / دليل المستخدم")
    
    # Language selector
    col1, col2, spacer = st.columns([1, 1, 8])
    with col1:
        if st.button("English", use_container_width=True):
            st.session_state["guide_lang"] = "en"
            st.rerun()
    with col2:
        if st.button("العربية", use_container_width=True):
            st.session_state["guide_lang"] = "ar"
            st.rerun()
    
    lang = st.session_state.get("guide_lang", "en")
    
    if lang == "ar":
        _render_arabic()
    else:
        _render_english()


def _render_english():
    st.markdown("---")
    
    st.markdown("""
    ### Welcome to the IR System! 
    
    This guide will walk you through using the Information Retrieval System step by step.
    The system provides a complete workflow for document indexing, search, and evaluation with 8 main steps.
    """)
    
    with st.expander("📋 **Overview**", expanded=True):
        st.markdown("""
        The IR System is a comprehensive information retrieval platform featuring:
        - Multiple indexing strategies (TF-IDF, BM25, Embeddings)
        - Hybrid retrieval methods (Serial, Parallel Fusion)
        - Query refinement techniques
        - Comprehensive evaluation metrics
        - Real-time system monitoring
        """)
    
    with st.expander("**1️⃣ Load Dataset** 📊"):
        st.markdown("""
        **Purpose**: Load a standard IR dataset for indexing and retrieval
        
        **Steps**:
        1. Navigate to the **Dataset** page from the sidebar
        2. Select a dataset from the dropdown (e.g., `beir/nfcorpus`, `cranfield`)
        3. Click **Load Dataset**
        4. Wait for confirmation that documents, queries, and qrels are loaded
        
        **Tips**:
        - Start with smaller datasets like `cranfield` (~1,400 docs)
        - `beir/nfcorpus` is good for medical domain (~3,600 docs)
        - Large datasets like `msmarco-passage` may take significant time
        """)
    
    with st.expander("**2️⃣ Upload to Database** 💾"):
        st.markdown("""
        **Purpose**: Persist loaded data to PostgreSQL for efficient retrieval
        
        **Steps**:
        1. After loading a dataset, click **Upload to Database**
        2. Monitor the upload progress
        3. Verify upload completion
        
        **Why?**: Database storage enables:
        - Fast document retrieval
        - Efficient query processing
        - Persistent storage across sessions
        """)
    
    with st.expander("**3️⃣ Preprocess Documents** ⚙️"):
        st.markdown("""
        **Purpose**: Prepare text for indexing by normalizing and transforming it
        
        **Configuration Options**:
        - **Tokenization**: spaCy (advanced) or Simple (fast)
        - **Lowercase**: Convert all text to lowercase
        - **Remove Punctuation**: Strip punctuation marks
        - **Remove Stopwords**: Filter common words (the, is, a, etc.)
        - **Morphological Processing**:
          - **Lemmatization**: Convert words to dictionary form (learning → learn)
          - **Stemming**: Aggressive suffix removal (machine → machin)
          - **Neither**: Keep original word forms
        - **Token Length**: Min/max character limits
        
        **Quick Presets**:
        - **Default (Academic)**: Balanced for research datasets
        - **Aggressive (Stemming)**: Maximum term reduction
        - **Minimal**: Preserves most original terms
        
        **Testing**: Use the **Preprocessing Lab** to test configurations before indexing
        """)
    
    with st.expander("**4️⃣ Build Index** 🗂️"):
        st.markdown("""
        **Purpose**: Create searchable indexes from preprocessed documents
        
        **Index Types**:
        
        **TF-IDF (Vector Space Model)**:
        - Classic term frequency-inverse document frequency
        - Good baseline method
        - Built automatically with inverted index
        
        **BM25**:
        - Probabilistic retrieval model
        - Parameters: k1 (term saturation), b (length normalization)
        - Typically superior to TF-IDF
        - Built from inverted index structure
        
        **Embeddings (Vector Index)**:
        - Dense vector representations
        - Enables semantic similarity search
        - Uses `sentence-transformers/all-MiniLM-L6-v2` model
        - Ideal for meaning-based queries
        
        **Steps**:
        1. Go to **Indexing** page
        2. Select preprocessing configuration
        3. Choose index type(s) to build
        4. Click **Build Index**
        5. Monitor build progress
        
        **Build Times** (approximate):
        - TF-IDF/BM25: ~2-3 minutes
        - Embeddings: ~10-20 minutes (GPU recommended for large datasets)
        
        **Note**: You can build multiple indexes and compare their performance
        """)
    
    with st.expander("**5️⃣ Perform Search** 🔍"):
        st.markdown("""
        **Purpose**: Retrieve relevant documents for a query
        
        **Retrieval Methods**:
        
        **TF-IDF (Vector Space Model)**:
        - Classic term-frequency based retrieval
        - Good baseline method
        
        **BM25**:
        - Probabilistic retrieval model
        - Often outperforms TF-IDF
        - Tune k1 (1.2-2.0) and b (0.75)
        
        **Embedding**:
        - Semantic similarity using dense vectors
        - Understands meaning beyond keywords
        - Good for paraphrase queries
        
        **Hybrid Serial** (Re-ranking Pipeline):
        1. First retriever gets candidate set (e.g., 1000 docs)
        2. Second retriever re-ranks candidates
        3. Returns top-k final results
        
        **Hybrid Parallel** (Fusion):
        - Retrieves from multiple methods simultaneously
        - Combines scores using fusion strategies:
          - **RRF**: Reciprocal Rank Fusion
          - **CombSUM**: Sum of normalized scores
          - **CombMNZ**: CombSUM × number of methods
          - **Weighted**: Custom weights per method
        
        **Steps**:
        1. Navigate to **Search** page
        2. Enter your query
        3. Select retrieval method
        4. Configure parameters
        5. Click **Search**
        6. Review ranked results
        """)
    
    with st.expander("**6️⃣ Refine Queries** 🔬 (Optional)"):
        st.markdown("""
        **Purpose**: Improve query quality through automatic enhancement
        
        **Techniques**:
        
        **Spell Correction**:
        - Fixes typos automatically
        - Configurable edit distance
        
        **Synonym Expansion**:
        - Adds related terms from WordNet
        - Configurable max synonyms per term
        
        **Query Suggestions**:
        - Generates alternative phrasings
        - Based on semantic similarity
        
        **Relevance Feedback**:
        - Learns from past queries
        - Suggests similar historical queries
        
        **Configuration**:
        - Navigate to **Query Refinement** page
        - Adjust parameters:
          - Max edit distance (1-3)
          - Confidence threshold (0-1)
          - Max synonyms (0-5)
          - History similarity threshold
        """)
    
    with st.expander("**7️⃣ Evaluate Performance** 📈"):
        st.markdown("""
        **Purpose**: Measure retrieval quality using standard IR metrics
        
        **Metrics**:
        
        **MAP (Mean Average Precision)**:
        - Overall ranking quality
        - Higher is better (0-1)
        
        **Recall**:
        - Proportion of relevant docs retrieved
        - Higher is better (0-1)
        
        **Precision@10**:
        - Precision in top 10 results
        - Higher is better (0-1)
        
        **nDCG (Normalized Discounted Cumulative Gain)**:
        - Ranking quality with graded relevance
        - Higher is better (0-1)
        
        **Steps**:
        1. Go to **Evaluation** page
        2. Select retrieval methods to compare
        3. Configure eval parameters (cutoff, queries)
        4. Click **Run Evaluation**
        5. Review metric tables and charts
        6. Compare methods
        
        **Tips**:
        - Run evaluation on a subset of queries first
        - BM25 typically outperforms TF-IDF
        - Hybrid methods often achieve best results
        - Embeddings excel at semantic queries
        """)
    
    with st.expander("**8️⃣ Monitor System Status** 🖥️"):
        st.markdown("""
        **Purpose**: Ensure all microservices are healthy and responsive
        
        **Services**:
        - **API Gateway**: Main entry point
        - **Preprocessing**: Text processing service
        - **Indexing**: Index building service
        - **Retrieval**: Search execution service
        - **Ranking**: Re-ranking service
        - **Query Refinement**: Query enhancement service
        - **Evaluation**: Metrics computation service
        
        **Features**:
        - Real-time health checks
        - Response time monitoring
        - Auto-refresh option
        - Service dependency graph
        
        **Troubleshooting**:
        - Red status: Service down or unreachable
        - Yellow status: Slow response
        - Green status: Healthy
        
        Navigate to **System Status** page to monitor.
        """)
    
    st.markdown("---")
    
    with st.expander("💡 **Quick Tips**"):
        st.markdown("""
        - **Start Small**: Use `cranfield` or `beir/nfcorpus` for learning
        - **Test Preprocessing**: Use Preprocessing Lab before building indexes
        - **Compare Methods**: Evaluate multiple retrieval strategies
        - **Monitor System**: Check System Status if operations fail
        - **Save Configurations**: Document which configs work best for your use case
        - **Hybrid Power**: Hybrid methods often outperform single methods
        """)
    
    with st.expander("🏗️ **System Architecture**"):
        st.markdown("""
        The IR System uses a **Service-Oriented Architecture (SOA)** with:
        - **7 Microservices**: Each handles a specific task
        - **API Gateway**: Routes requests to appropriate services
        - **PostgreSQL Database**: Stores documents and metadata
        - **Streamlit Frontend**: User interface
        - **Docker**: Containerized deployment
        
        All services communicate via REST APIs for modularity and scalability.
        """)


def _render_arabic():
    st.markdown('<div dir="rtl">', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
    ### مرحباً بك في نظام استرجاع المعلومات!
    
    هذا الدليل سيرشدك خطوة بخطوة لاستخدام نظام استرجاع المعلومات.
    يوفر النظام سير عمل كامل لفهرسة المستندات والبحث والتقييم بـ 8 خطوات رئيسية.
    """)
    
    with st.expander("📋 **نظرة عامة**", expanded=True):
        st.markdown("""
        نظام استرجاع المعلومات هو منصة شاملة توفر:
        - استراتيجيات فهرسة متعددة (TF-IDF, BM25, Embeddings)
        - طرق استرجاع هجينة (Serial, Parallel Fusion)
        - تقنيات تحسين الاستعلامات
        - مقاييس تقييم شاملة
        - مراقبة النظام في الوقت الفعلي
        """)
    
    with st.expander("**1️⃣ تحميل مجموعة البيانات** 📊"):
        st.markdown("""
        **الهدف**: تحميل مجموعة بيانات قياسية للفهرسة والاسترجاع
        
        **الخطوات**:
        1. انتقل إلى صفحة **Dataset** من الشريط الجانبي
        2. اختر مجموعة بيانات من القائمة (مثل `beir/nfcorpus` أو `cranfield`)
        3. انقر على **Load Dataset**
        4. انتظر تأكيد تحميل المستندات والاستعلامات وال qrels
        
        **نصائح**:
        - ابدأ بمجموعات بيانات صغيرة مثل `cranfield` (~1,400 مستند)
        - `beir/nfcorpus` جيد للمجال الطبي (~3,600 مستند)
        - المجموعات الكبيرة مثل `msmarco-passage` قد تستغرق وقتاً طويلاً
        """)
    
    with st.expander("**2️⃣ رفع إلى قاعدة البيانات** 💾"):
        st.markdown("""
        **الهدف**: حفظ البيانات المحملة في PostgreSQL للاسترجاع الفعال
        
        **الخطوات**:
        1. بعد تحميل مجموعة البيانات، انقر على **Upload to Database**
        2. راقب تقدم الرفع
        3. تحقق من اكتمال الرفع
        
        **لماذا؟** تخزين قاعدة البيانات يتيح:
        - استرجاع سريع للمستندات
        - معالجة فعالة للاستعلامات
        - تخزين دائم عبر الجلسات
        """)
    
    with st.expander("**3️⃣ معالجة المستندات مسبقاً** ⚙️"):
        st.markdown("""
        **الهدف**: تحضير النص للفهرسة عن طريق تطبيعه وتحويله
        
        **خيارات الإعداد**:
        - **التقسيم (Tokenization)**: spaCy (متقدم) أو Simple (سريع)
        - **الأحرف الصغيرة (Lowercase)**: تحويل كل النص لأحرف صغيرة
        - **إزالة علامات الترقيم**: حذف علامات الترقيم
        - **إزالة الكلمات الشائعة**: تصفية الكلمات الشائعة (the, is, a, etc.)
        - **المعالجة الصرفية**:
          - **Lemmatization**: تحويل الكلمات للشكل القاموسي (learning → learn)
          - **Stemming**: إزالة عدوانية للاحقات (machine → machin)
          - **Neither**: الحفاظ على الشكل الأصلي للكلمات
        - **طول الرمز**: حدود أدنى/أقصى للأحرف
        
        **إعدادات سريعة**:
        - **Default (Academic)**: متوازن لمجموعات البحث العلمي
        - **Aggressive (Stemming)**: أقصى تقليل للمصطلحات
        - **Minimal**: يحافظ على معظم المصطلحات الأصلية
        
        **الاختبار**: استخدم **Preprocessing Lab** لاختبار الإعدادات قبل الفهرسة
        """)
    
    with st.expander("**4️⃣ بناء الفهرس** 🗂️"):
        st.markdown("""
        **الهدف**: إنشاء فهارس قابلة للبحث من المستندات المعالجة
        
        **أنواع الفهارس**:
        
        **TF-IDF (نموذج الفضاء الشعاعي)**:
        - استرجاع كلاسيكي مبني على تردد المصطلحات
        - طريقة أساسية جيدة
        - يُبنى تلقائياً مع الفهرس المعكوس
        
        **BM25**:
        - نموذج استرجاع احتمالي
        - المعاملات: k1 (تشبع المصطلح)، b (تطبيع الطول)
        - عادة أفضل من TF-IDF
        - يُبنى من بنية الفهرس المعكوس
        
        **Embeddings (الفهرس الشعاعي)**:
        - تمثيلات شعاعية كثيفة
        - يمكّن البحث بالتشابه الدلالي
        - يستخدم نموذج `sentence-transformers/all-MiniLM-L6-v2`
        - مثالي للاستعلامات المبنية على المعنى
        
        **الخطوات**:
        1. اذهب إلى صفحة **Indexing**
        2. اختر إعداد المعالجة المسبقة
        3. اختر نوع/أنواع الفهرس المراد بناؤها
        4. انقر على **Build Index**
        5. راقب تقدم البناء
        
        **أوقات البناء** (تقريبية):
        - TF-IDF/BM25: ~2-3 دقائق
        - Embeddings: ~10-20 دقيقة (GPU موصى به للمجموعات الكبيرة)
        
        **ملاحظة**: يمكنك بناء فهارس متعددة ومقارنة أدائها
        """)
    
    with st.expander("**5️⃣ إجراء البحث** 🔍"):
        st.markdown("""
        **الهدف**: استرجاع المستندات ذات الصلة لاستعلام معين
        
        **طرق الاسترجاع**:
        
        **TF-IDF (نموذج الفضاء الشعاعي)**:
        - استرجاع كلاسيكي مبني على تردد المصطلحات
        - طريقة أساسية جيدة
        
        **BM25**:
        - نموذج استرجاع احتمالي
        - عادة يتفوق على TF-IDF
        - اضبط k1 (1.2-2.0) و b (0.75)
        
        **Embedding**:
        - تشابه دلالي باستخدام متجهات كثيفة
        - يفهم المعنى خارج الكلمات المفتاحية
        - جيد لاستعلامات إعادة الصياغة
        
        **Hybrid Serial** (خط إعادة ترتيب):
        1. المسترجع الأول يحصل على مجموعة مرشحة (مثلاً 1000 مستند)
        2. المسترجع الثاني يعيد ترتيب المرشحين
        3. يعيد النتائج النهائية الأعلى
        
        **Hybrid Parallel** (الدمج):
        - يسترجع من طرق متعددة في نفس الوقت
        - يدمج النتائج باستخدام استراتيجيات الدمج:
          - **RRF**: دمج الترتيب المتبادل
          - **CombSUM**: مجموع النتائج المطبعة
          - **CombMNZ**: CombSUM × عدد الطرق
          - **Weighted**: أوزان مخصصة لكل طريقة
        
        **الخطوات**:
        1. انتقل إلى صفحة **Search**
        2. أدخل استعلامك
        3. اختر طريقة الاسترجاع
        4. اضبط المعاملات
        5. انقر على **Search**
        6. راجع النتائج المرتبة
        """)
    
    with st.expander("**6️⃣ تحسين الاستعلامات** 🔬 (اختياري)"):
        st.markdown("""
        **الهدف**: تحسين جودة الاستعلام من خلال التحسين التلقائي
        
        **التقنيات**:
        
        **تصحيح الإملاء**:
        - يصحح الأخطاء الإملائية تلقائياً
        - مسافة تحرير قابلة للضبط
        
        **توسيع المرادفات**:
        - يضيف مصطلحات مرتبطة من WordNet
        - حد أقصى قابل للضبط للمرادفات لكل مصطلح
        
        **اقتراحات الاستعلام**:
        - يولد صياغات بديلة
        - مبني على التشابه الدلالي
        
        **التغذية الراجعة على الصلة**:
        - يتعلم من الاستعلامات السابقة
        - يقترح استعلامات تاريخية مشابهة
        
        **الإعداد**:
        - انتقل إلى صفحة **Query Refinement**
        - اضبط المعاملات:
          - مسافة التحرير القصوى (1-3)
          - عتبة الثقة (0-1)
          - حد أقصى للمرادفات (0-5)
          - عتبة التشابه التاريخي
        """)
    
    with st.expander("**7️⃣ تقييم الأداء** 📈"):
        st.markdown("""
        **الهدف**: قياس جودة الاسترجاع باستخدام مقاييس IR القياسية
        
        **المقاييس**:
        
        **MAP (متوسط الدقة المتوسطة)**:
        - جودة الترتيب الإجمالية
        - أعلى أفضل (0-1)
        
        **Recall (الاستدعاء)**:
        - نسبة المستندات ذات الصلة المستردة
        - أعلى أفضل (0-1)
        
        **Precision@10 (الدقة عند 10)**:
        - الدقة في أعلى 10 نتائج
        - أعلى أفضل (0-1)
        
        **nDCG (مكسب تراكمي مخصوم مطبع)**:
        - جودة الترتيب مع صلة متدرجة
        - أعلى أفضل (0-1)
        
        **الخطوات**:
        1. اذهب إلى صفحة **Evaluation**
        2. اختر طرق الاسترجاع للمقارنة
        3. اضبط معاملات التقييم (cutoff، queries)
        4. انقر على **Run Evaluation**
        5. راجع جداول المقاييس والرسوم البيانية
        6. قارن الطرق
        
        **نصائح**:
        - نفذ التقييم على مجموعة فرعية من الاستعلامات أولاً
        - BM25 عادة يتفوق على TF-IDF
        - الطرق الهجينة غالباً تحقق أفضل النتائج
        - Embeddings يتفوق في الاستعلامات الدلالية
        """)
    
    with st.expander("**8️⃣ مراقبة حالة النظام** 🖥️"):
        st.markdown("""
        **الهدف**: التأكد من أن جميع الخدمات المصغرة صحية ومستجيبة
        
        **الخدمات**:
        - **API Gateway**: نقطة الدخول الرئيسية
        - **Preprocessing**: خدمة معالجة النصوص
        - **Indexing**: خدمة بناء الفهرس
        - **Retrieval**: خدمة تنفيذ البحث
        - **Ranking**: خدمة إعادة الترتيب
        - **Query Refinement**: خدمة تحسين الاستعلامات
        - **Evaluation**: خدمة حساب المقاييس
        
        **المميزات**:
        - فحوصات صحية في الوقت الفعلي
        - مراقبة وقت الاستجابة
        - خيار التحديث التلقائي
        - رسم بياني لتبعيات الخدمة
        
        **استكشاف الأخطاء**:
        - الحالة الحمراء: الخدمة متوقفة أو غير قابلة للوصول
        - الحالة الصفراء: استجابة بطيئة
        - الحالة الخضراء: صحية
        
        انتقل إلى صفحة **System Status** للمراقبة.
        """)
    
    st.markdown("---")
    
    with st.expander("💡 **نصائح سريعة**"):
        st.markdown("""
        - **ابدأ صغيراً**: استخدم `cranfield` أو `beir/nfcorpus` للتعلم
        - **اختبر المعالجة المسبقة**: استخدم Preprocessing Lab قبل بناء الفهارس
        - **قارن الطرق**: قيّم استراتيجيات استرجاع متعددة
        - **راقب النظام**: تحقق من System Status إذا فشلت العمليات
        - **احفظ الإعدادات**: وثّق الإعدادات التي تعمل بشكل أفضل لحالتك
        - **قوة الهجين**: الطرق الهجينة غالباً تتفوق على الطرق الفردية
        """)
    
    with st.expander("🏗️ **بنية النظام**"):
        st.markdown("""
        يستخدم نظام IR **بنية موجهة نحو الخدمات (SOA)** مع:
        - **7 خدمات مصغرة**: كل واحدة تتعامل مع مهمة محددة
        - **API Gateway**: يوجه الطلبات إلى الخدمات المناسبة
        - **قاعدة بيانات PostgreSQL**: تخزن المستندات والبيانات الوصفية
        - **Streamlit Frontend**: واجهة المستخدم
        - **Docker**: نشر في حاويات
        
        جميع الخدمات تتواصل عبر REST APIs للنمطية وقابلية التوسع.
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

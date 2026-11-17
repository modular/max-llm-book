// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded affix "><a href="introduction.html">Introduction</a></li><li class="chapter-item expanded "><a href="ch01.html">1. Foundations</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="01_01_run_and_serve.html">1.1 Run and serve an LLM</a></li><li class="chapter-item expanded "><a href="01_02_tensors_and_eager.html">1.2 Tensors and eager execution</a></li><li class="chapter-item expanded "><a href="01_03_tokens_and_weights.html">1.3 Tokens, embeddings, and weights</a></li><li class="chapter-item expanded "><a href="01_04_building_blocks.html">1.4 Building blocks</a></li><li class="chapter-item expanded "><a href="01_05_attention.html">1.5 Attention</a></li><li class="chapter-item expanded "><a href="01_06_full_model.html">1.6 The full model</a></li><li class="chapter-item expanded "><a href="01_07_sampling.html">1.7 Generation and sampling</a></li><li class="chapter-item expanded "><a href="01_08_kv_cache.html">1.8 The KV cache</a></li></ol></li><li class="chapter-item expanded "><a href="ch02.html">2. Distributed tensors</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="02_01_distributed_tensors.html">2.1 Distributed tensors</a></li><li class="chapter-item expanded "><a href="02_02_tensor_parallel.html">2.2 A tensor-parallel model</a></li></ol></li><li class="chapter-item expanded "><a href="ch03.html">3. Custom ops in Mojo</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="03_01_custom_ops.html">3.1 Your first custom op</a></li><li class="chapter-item expanded "><a href="03_02_graph_api.html">3.2 Under the hood: the graph API</a></li></ol></li><li class="chapter-item expanded "><a href="ch04.html">4. Testing and benchmarking</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="04_01_layer_tests.html">4.1 Testing layers against references</a></li><li class="chapter-item expanded "><a href="04_02_logit_verification.html">4.2 Verifying a model&#39;s logits</a></li><li class="chapter-item expanded "><a href="04_03_kernel_benchmarks.html">4.3 Benchmarking kernels and layers</a></li><li class="chapter-item expanded "><a href="04_04_profiling.html">4.4 Profiling</a></li></ol></li><li class="chapter-item expanded "><a href="ch05.html">5. A serving engine</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="05_01_request_lifecycle.html">5.1 The life of a request</a></li><li class="chapter-item expanded "><a href="05_02_continuous_batching.html">5.2 Continuous batching</a></li><li class="chapter-item expanded "><a href="05_03_paged_kv_cache.html">5.3 A paged KV cache</a></li><li class="chapter-item expanded "><a href="05_04_prefix_caching.html">5.4 Prefix caching</a></li><li class="chapter-item expanded "><a href="05_05_speculative_decoding.html">5.5 Speculative decoding</a></li></ol></li><li class="chapter-item expanded "><a href="ch06.html">6. Bigger models</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="06_01_qwen3.html">6.1 Qwen3: normalized queries and keys</a></li><li class="chapter-item expanded "><a href="06_02_gemma3.html">6.2 Gemma 3: sliding-window attention</a></li><li class="chapter-item expanded "><a href="06_03_mixture_of_experts.html">6.3 Mixture of experts</a></li></ol></li><li class="chapter-item expanded "><a href="ch07.html">7. Production serving</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="07_01_serve_your_model.html">7.1 Serve your architecture</a></li><li class="chapter-item expanded "><a href="07_02_benchmark_serving.html">7.2 Benchmark a server</a></li><li class="chapter-item expanded "><a href="07_03_structured_output.html">7.3 Structured output and tools</a></li><li class="chapter-item expanded "><a href="07_04_modular_cloud.html">7.4 Modular&#39;s cloud</a></li></ol></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split("#")[0].split("?")[0];
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);

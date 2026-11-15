#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include "tensor.hpp"

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("   FAIL line %d: %s\n", __LINE__, #cond); ++failures; return; } } while (0)

static void fill_arange(Tensor& t) {   // fills t (assumed freshly created, contiguous) with 0,1,2,...
    int64_t n = t.numel();
    float* p = const_cast<float*>(t.storage_ptr()) + t.offset();
    for (int64_t i = 0; i < n; ++i) p[i] = static_cast<float>(i);
}

static void test_strides() {
    Tensor t({3, 4});
    CHECK(t.strides() == Shape({4, 1}));
    CHECK(t.numel() == 12 && t.offset() == 0 && t.is_contiguous());
    Tensor u({2, 3, 4});
    CHECK(u.strides() == Shape({12, 4, 1}));
}
static void test_at() {
    Tensor t({3, 4}); fill_arange(t);
    CHECK(t.at({0, 0}) == 0.f && t.at({1, 2}) == 6.f && t.at({2, 3}) == 11.f);
    t.at({1, 1}) = 99.f;
    CHECK(t.at({1, 1}) == 99.f);
}
static void test_at_bounds() {
    Tensor t({3, 4});
    bool threw = false;
    try { t.at({3, 0}); } catch (const std::out_of_range&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { t.at({0}); } catch (const std::out_of_range&) { threw = true; }
    CHECK(threw);
}
static void test_transpose_is_a_view() {
    Tensor t({3, 4}); fill_arange(t);
    Tensor tt = t.transpose(0, 1);
    CHECK(tt.shape() == Shape({4, 3}));
    CHECK(tt.strides() == Shape({1, 4}));
    CHECK(tt.storage_ptr() == t.storage_ptr());          // no copy
    CHECK(tt.at({2, 1}) == t.at({1, 2}));
    tt.at({0, 2}) = -5.f;                                // write through the view...
    CHECK(t.at({2, 0}) == -5.f);                         // ...changes the original
    CHECK(!tt.is_contiguous());
}
static void test_transpose_3d() {
    Tensor t({2, 3, 4}); fill_arange(t);
    Tensor p = t.transpose(0, 2);
    CHECK(p.shape() == Shape({4, 3, 2}) && p.strides() == Shape({1, 4, 12}));
    CHECK(p.at({3, 2, 1}) == t.at({1, 2, 3}));
}
static void test_select() {
    Tensor t({3, 4}); fill_arange(t);
    Tensor row = t.select(0, 1);                         // t[1, :]
    CHECK(row.shape() == Shape({4}) && row.strides() == Shape({1}) && row.offset() == 4);
    CHECK(row.at({2}) == 6.f);
    Tensor col = t.select(1, 2);                         // t[:, 2]
    CHECK(col.shape() == Shape({3}) && col.strides() == Shape({4}) && col.offset() == 2);
    CHECK(col.at({1}) == 6.f && col.at({2}) == 10.f);
    bool threw = false;
    try { t.select(0, 3); } catch (const std::out_of_range&) { threw = true; }
    CHECK(threw);
}
static void test_contiguous_copy() {
    Tensor t({3, 4}); fill_arange(t);
    Tensor tt = t.transpose(0, 1);
    Tensor c = tt.contiguous();
    CHECK(c.is_contiguous());
    CHECK(c.shape() == Shape({4, 3}) && c.strides() == Shape({3, 1}) && c.offset() == 0);
    CHECK(c.storage_ptr() != t.storage_ptr());           // fresh storage
    for (int64_t i = 0; i < 4; ++i)
        for (int64_t j = 0; j < 3; ++j) CHECK(c.at({i, j}) == t.at({j, i}));
    c.at({0, 0}) = 123.f;
    CHECK(t.at({0, 0}) == 0.f);                          // copy is independent
}
static void test_contiguous_of_select() {
    Tensor t({3, 4}); fill_arange(t);
    Tensor col = t.select(1, 1).contiguous();            // [1, 5, 9]
    CHECK(col.shape() == Shape({3}) && col.at({0}) == 1.f && col.at({1}) == 5.f && col.at({2}) == 9.f);
}
static void test_reshape() {
    Tensor t({3, 4}); fill_arange(t);
    Tensor r = t.reshape({2, 6});
    CHECK(r.shape() == Shape({2, 6}) && r.strides() == Shape({6, 1}));
    CHECK(r.storage_ptr() == t.storage_ptr());           // reshape of contiguous = view
    CHECK(r.at({1, 0}) == 6.f);
    bool threw = false;
    try { t.transpose(0, 1).reshape({12}); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);                                        // must contiguous() first
    Tensor ok = t.transpose(0, 1).contiguous().reshape({12});
    CHECK(ok.at({1}) == 4.f);                            // transposed order: 0,4,8,1,...
}
static void test_shared_lifetime() {
    Tensor view = [] {
        Tensor t({2, 2});
        t.at({1, 1}) = 7.f;
        return t.transpose(0, 1);                        // t dies here; storage must live on
    }();
    CHECK(view.at({1, 1}) == 7.f);
}

int main() {
    struct T { const char* name; std::function<void()> fn; };
    std::vector<T> tests = {
        {"strides", test_strides}, {"at", test_at}, {"at_bounds", test_at_bounds},
        {"transpose_is_a_view", test_transpose_is_a_view}, {"transpose_3d", test_transpose_3d},
        {"select", test_select}, {"contiguous_copy", test_contiguous_copy},
        {"contiguous_of_select", test_contiguous_of_select}, {"reshape", test_reshape},
        {"shared_lifetime", test_shared_lifetime}};
    int passed = 0;
    for (auto& t : tests) {
        int before = failures;
        try { t.fn(); }
        catch (const NotImplemented&) { std::printf("TODO  %-22s\n", t.name); ++failures; continue; }
        catch (const std::exception& e) { std::printf("ERROR %-22s (%s)\n", t.name, e.what()); ++failures; continue; }
        if (failures == before) { std::printf("PASS  %s\n", t.name); ++passed; }
        else std::printf("FAIL  %s\n", t.name);
    }
    std::printf("\n%d/%zu passed\n", passed, tests.size());
    return failures ? 1 : 0;
}

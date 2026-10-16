// Copyright(C) 2026 InfiniFlow, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

// Dictionary forms of inflected Slovak words, for the forms the light stemmer
// cannot reduce on its own ("ľudí" -> "ľudia", "psa" -> "pes", "som" -> "byť").
//
// The table is resource/slovak/lemma.tsv: one "form<TAB>lemma" line per
// lowercase form, sorted bytewise by form. Infinity's
// tools/generate_slovak_lemma_table.py derives it from the hunspell-sk
// dictionary, and it carries that dictionary's licence (MPL-2.0) next to it.
// Mirrors _SlovakLemmatizer in Infinity's rag_tokenizer.py.
class SlovakLemmatizer {
public:
    static constexpr const char *TABLE_PATH = "slovak/lemma.tsv";

    // The table under a resource directory, read once per process and shared
    // by every analyzer; nullptr when the resource directory has none, in
    // which case Slovak is stemmed without a lemma lookup.
    static const SlovakLemmatizer *Shared(const std::string &resource_path) {
        static std::mutex mutex;
        static std::map<std::string, std::unique_ptr<SlovakLemmatizer>> tables;
        std::lock_guard<std::mutex> lock(mutex);
        auto [it, inserted] = tables.try_emplace(resource_path);
        if (inserted) {
            std::string path = resource_path + "/" + TABLE_PATH;
            std::ifstream file(path, std::ios::binary);
            if (file) {
                std::ostringstream data;
                data << file.rdbuf();
                it->second.reset(new SlovakLemmatizer(std::move(data).str()));
            } else {
                printf("Slovak lemma table not found, stemming without it: %s\n", path.c_str());
            }
        }
        return it->second.get();
    }

    // The lemma of a lowercase form, or the form itself when the table has no
    // entry for it.
    std::string_view Lemmatize(std::string_view form) const {
        auto it = std::lower_bound(lines_.begin(), lines_.end(), form, [this](uint32_t line, std::string_view key) {
            return Form(line) < key;
        });
        if (it == lines_.end() || Form(*it) != form) {
            return form;
        }
        size_t lemma = *it + form.size() + 1;
        return std::string_view(data_).substr(lemma, data_.find('\n', lemma) - lemma);
    }

private:
    explicit SlovakLemmatizer(std::string data) : data_(std::move(data)) {
        for (size_t line = 0; line < data_.size();) {
            size_t end = data_.find('\n', line);
            if (end == std::string::npos) {
                data_ += '\n';
                end = data_.size() - 1;
            }
            if (data_.find('\t', line) < end) {
                lines_.push_back(static_cast<uint32_t>(line));
            }
            line = end + 1;
        }
    }

    std::string_view Form(uint32_t line) const {
        return std::string_view(data_).substr(line, data_.find('\t', line) - line);
    }

    std::string data_;
    std::vector<uint32_t> lines_; // offset of each line, in the table's (bytewise) order
};

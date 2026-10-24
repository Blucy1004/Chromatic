#pragma once
#include "search.hpp"
#include <iosfwd>
#include <string>
namespace zero {
void write_telemetry_json(std::ostream& out, const SearchTelemetry& data);
std::string search_result_json(const SearchResult& result);
}

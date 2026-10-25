#include "zero/pvs.hpp"
#include "zero/rrn.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

using namespace zero;
static void require(bool ok,const std::string& text){if(!ok)throw std::runtime_error(text);}
static bool has_move(const std::vector<Move>& moves,const std::string& uci){return std::any_of(moves.begin(),moves.end(),[&](const Move&m){return m.uci()==uci;});}
static void assert_legal(const Board& root,const PvsResult& result,const char* label){
  require(result.has_move,std::string(label)+": missing bestmove");
  auto legal=root.legal();
  require(std::find(legal.begin(),legal.end(),result.best)!=legal.end(),std::string(label)+": bestmove is illegal");
  require(!result.pv.empty()&&result.pv.front()==result.best,std::string(label)+": PV does not start with bestmove");
}
int main(){try {
  std::atomic<bool> stop=false;
  Board start=Board::from_fen(START_FEN);
  const std::string original=start.fen();
  require(perft(start,3)==8902,"startpos perft(3)");
  require(start.fen()==original,"perft mutated caller board");

  PvsSearch engine(8);
  PvsLimits limits; limits.max_depth=3; limits.max_nodes=200000; limits.tt_megabytes=8;
  auto ordinary=engine.run(start,limits,stop);
  require(ordinary.completed&&ordinary.completed_depth>=1,"ordinary search completed no iteration");
  assert_legal(start,ordinary,"start position");
  require(start.fen()==original,"PVS mutated caller board");

  // Board transitions: the search must operate on copies and preserve all
  // caller state; direct transition fixtures check castling, EP and promotion.
  Board castle=Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
  const auto castle_before=castle.fen(); const auto castle_key=castle.key(); const auto castle_history=castle.history;
  castle.play("e1g1");
  require(castle.squares[6]==6&&castle.squares[5]==4&&castle.squares[7]==0,"castling king/rook transition");
  require(castle.turn==-1&&castle.castle==12&&castle.fullmove==1&&castle.halfmove==1,"castling rights/turn/clocks");
  require(castle.history.size()==castle_history.size()+1&&castle.key()!=castle_key,"castling history/hash");
  require(castle_before=="r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1","castling fixture changed unexpectedly");

  Board ep=Board::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 7 12");
  const auto ep_before=ep.fen(); const auto ep_history=ep.history;
  ep.play("e5d6");
  require(ep.squares[43]==1&&ep.squares[35]==0&&ep.squares[36]==0,"en-passant captured/capturing pawn squares");
  require(ep.ep==-1&&ep.halfmove==0&&ep.fullmove==12&&ep.turn==-1,"en-passant clocks/turn/target");
  require(ep.history.size()==1&&ep.history.size()<=ep_history.size(),"pawn capture did not reset history window");
  require(ep_before=="4k3/8/8/3pP3/8/8/8/4K3 w - d6 7 12","EP fixture changed unexpectedly");

  Board promotion=Board::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 9 4");
  promotion.play("a7a8q");
  require(promotion.squares[56]==5&&promotion.squares[48]==0,"queen promotion piece placement");
  require(promotion.turn==-1&&promotion.halfmove==0&&promotion.fullmove==4&&promotion.history.size()==1,"promotion clocks/history");
  // A legal in-check response must be generated; the chosen evasion must end check.
  Board checked=Board::from_fen("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1");
  const int checked_side=checked.turn;
  require(checked.check(checked_side),"check-evasion fixture is not check");
  auto evasions=checked.legal(); require(!evasions.empty(),"check-evasion fixture has no legal move");
  limits.max_depth=2; limits.max_nodes=100000;
  auto evasion=engine.run(checked,limits,stop); assert_legal(checked,evasion,"check evasion");
  Board escaped=checked; escaped.make(evasion.best);
  require(!escaped.check(checked_side),"search selected a move that failed to evade check");

  // Mate distance is encoded near the reserved mate score and the move verifies.
  Board mate=Board::from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1");
  limits.max_depth=3; limits.max_nodes=200000;
  auto mate_result=engine.run(mate,limits,stop); assert_legal(mate,mate_result,"mate in one");
  Board after_mate=mate; after_mate.make(mate_result.best); float terminal_value=0;
  require(after_mate.terminal(after_mate.legal(),terminal_value)&&terminal_value==-1,"best move did not deliver checkmate");
  require(mate_result.score_cp>PVS_MATE_SCORE-100,"mate score did not preserve mate-distance range");

  // Tactical special moves remain legal candidates and the Board caller is unchanged.
  for(const auto& fixture:std::vector<std::pair<std::string,std::string>>{
      {"1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1","a7a8"},
      {"r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1","e1g1"},
      {"4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1","e5d6"}}){
    Board b=Board::from_fen(fixture.first); const auto before=b.fen(); auto legal=b.legal();
    bool found=std::any_of(legal.begin(),legal.end(),[&](const Move&m){return m.uci().rfind(fixture.second,0)==0;});
    require(found,"special-move fixture omitted expected promotion/castle/EP candidate");
    if(fixture.second=="a7a8")require(std::count_if(legal.begin(),legal.end(),[](const Move&m){return m.promo!=0;})>=4,"promotion fixture lost underpromotion options");
    if(fixture.second=="e1g1")require(has_move(legal,"e1c1"),"queenside castling candidate missing");
    PvsSearch local(2); PvsLimits short_limits; short_limits.max_depth=1; short_limits.max_nodes=10000;
    auto r=local.run(b,short_limits,stop); assert_legal(b,r,"special move position");
    require(b.fen()==before,"special-move search mutated caller board");
  }

  Board fifty=Board::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 100 1");
  float draw_value=7.0f; auto fifty_moves=fifty.legal();
  require(fifty.terminal(fifty_moves,draw_value)&&draw_value==0.0f,"50-move root is not a draw");
  auto fifty_result=engine.run(fifty,limits,stop);
  require(fifty_result.terminal&&!fifty_result.has_move,"search did not preserve 50-move terminal");

  Board repeated=Board::from_fen(START_FEN);
  for(int n=0;n<2;++n)for(const char* uci:{"g1f3","g8f6","f3g1","f6g8"})repeated.play(uci);
  require(repeated.repetitions()>=3,"repetition fixture not threefold");
  draw_value=1.0f; require(repeated.terminal(repeated.legal(),draw_value)&&draw_value==0.0f,"threefold root is not a draw");
  auto repetition_result=engine.run(repeated,limits,stop);
  require(repetition_result.terminal&&!repetition_result.has_move,"search did not preserve repetition terminal");

  // TT must return the same fully searched value and count verified hits on reuse.
  Board tt_position=Board::from_fen(START_FEN); PvsSearch tt_engine(4);
  PvsLimits tt_limits; tt_limits.max_depth=3; tt_limits.max_nodes=250000; tt_limits.tt_megabytes=4;
  auto cold=tt_engine.run(tt_position,tt_limits,stop); auto warm=tt_engine.run(tt_position,tt_limits,stop);
  require(cold.completed&&warm.completed,"TT fixture did not finish requested iterative search");
  require(cold.score_cp==warm.score_cp&&cold.best==warm.best,"TT reuse changed fixed-depth result");
  require(warm.tt_hits>0,"TT reuse did not report a hit");

  // A pre-raised stop flag must return promptly without claiming a completed depth.
  std::atomic<bool> already_stopped=true; PvsSearch stopped_engine(2);
  PvsLimits stopped_limits; stopped_limits.max_depth=20; stopped_limits.max_nodes=1000000;
  auto stopped=stopped_engine.run(start,stopped_limits,already_stopped);
  require(!stopped.completed||stopped.completed_depth==0,"pre-stopped search claimed an iteration");
  require(stopped.nodes<=stopped_limits.max_nodes,"pre-stopped search exceeded node budget");

  // Exercise asynchronous stop and wall-clock deadline without asserting a
  // machine-dependent exact depth or elapsed time.
  PvsSearch async_engine(2); PvsLimits async_limits; async_limits.max_depth=40; async_limits.max_nodes=10000000;
  std::atomic<bool> async_stop=false;
  std::thread stopper([&](){std::this_thread::sleep_for(std::chrono::milliseconds(5));async_stop.store(true);});
  auto async_result=async_engine.run(start,async_limits,async_stop); stopper.join();
  require(async_result.completed_depth<async_limits.max_depth,"asynchronous stop did not interrupt the deep search");
  require(async_result.nodes<=async_limits.max_nodes,"asynchronously stopped search exceeded node cap");

  PvsSearch timed_engine(2); PvsLimits timed_limits; timed_limits.max_depth=40; timed_limits.max_nodes=10000000; timed_limits.milliseconds=2;
  auto timed_result=timed_engine.run(start,timed_limits,stop);
  require(timed_result.completed_depth<timed_limits.max_depth,"wall-clock deadline did not stop iterative deepening");
  require(timed_result.nodes<=timed_limits.max_nodes,"timed search exceeded node cap");
  // Hard node budget is enforced even when quiescence is needed.
  PvsSearch bounded_engine(2); PvsLimits bounded; bounded.max_depth=20; bounded.max_nodes=256;
  auto bounded_result=bounded_engine.run(start,bounded,stop);
  require(bounded_result.nodes<=bounded.max_nodes,"search exceeded hard node limit");
  require(bounded_result.completed_depth<bounded.max_depth,"small node budget unexpectedly completed all iterations");

  // RRN is a deterministic, explicitly untrained reference ranker. It emits
  // one finite advisory score per legal move, and serializes versioned weights.
  auto rrn_a=RrnRanker::reference(2026), rrn_b=RrnRanker::reference(2026);
  auto start_legal=start.legal(); auto scores_a=rrn_a.scores(start,start_legal); auto scores_b=rrn_b.rank(start,start_legal);
  require(!rrn_a.learned(),"deterministic reference weights mislabeled learned");
  require(rrn_a.provenance().find("reference")!=std::string::npos,"RRN reference provenance missing");
  require(scores_a.size()==start_legal.size()&&scores_a==scores_b,"RRN reference scores are not deterministic/aligned");
  for(float score:scores_a)require(std::isfinite(score),"RRN score is non-finite");
  const auto score_bounds=std::minmax_element(scores_a.begin(),scores_a.end());
  require(score_bounds.first!=scores_a.end()&&*score_bounds.first<*score_bounds.second,"RRN reference scores do not distinguish legal moves");
  RrnOrderCache rrn_cache(rrn_a,1);
  const auto cached_first=rrn_cache.rank(start,start_legal);
  const auto cached_second=rrn_cache.rank(start,start_legal);
  require(cached_first==scores_a&&cached_second==scores_a,"RRN cache changed score alignment or values");
  require(rrn_cache.calls()==2&&rrn_cache.hits()==1,"RRN cache failed to reuse an exact repeated position");
  Board halfmove_variant=start;halfmove_variant.halfmove=1;
  const auto halfmove_legal=halfmove_variant.legal();
  (void)rrn_cache.rank(halfmove_variant,halfmove_legal);
  require(rrn_cache.calls()==3&&rrn_cache.hits()==1,"RRN cache reused across a feature-changing halfmove state");
  (void)rrn_cache.rank(start,start_legal);
  require(rrn_cache.calls()==4&&rrn_cache.hits()==1,"RRN direct-mapped cache failed to recompute after collision/eviction");
  PvsSearch ordered_engine(4); PvsLimits ordered_limits; ordered_limits.max_depth=2; ordered_limits.max_nodes=100000;
  size_t ordering_calls=0;
  ordered_limits.ordering_hint=[&](const Board& b,const std::vector<Move>& moves){++ordering_calls;return rrn_a.rank(b,moves);};
  auto ordered=ordered_engine.run(start,ordered_limits,stop); assert_legal(start,ordered,"RRN advisory ordering");
  require(ordering_calls>0,"PVS did not invoke the RRN ordering callback");
  PvsSearch gated_engine(2); PvsLimits gated_limits; gated_limits.max_depth=2; gated_limits.max_nodes=100000; gated_limits.ordering_hint_min_depth=3;
  size_t gated_calls=0; gated_limits.ordering_hint=[&](const Board& b,const std::vector<Move>& moves){++gated_calls;return rrn_a.rank(b,moves);};
  auto gated_shallow=gated_engine.run(start,gated_limits,stop); assert_legal(start,gated_shallow,"depth-gated RRN shallow search");
  require(gated_calls==0,"RRN depth gate ran below its configured threshold");
  gated_limits.max_depth=3; auto gated_deep=gated_engine.run(start,gated_limits,stop); assert_legal(start,gated_deep,"depth-gated RRN deep search");
  require(gated_calls>0,"RRN depth gate never ran at its configured threshold");

  PvsSearch nan_hint_engine(2); PvsLimits nan_hint_limits; nan_hint_limits.max_depth=1; nan_hint_limits.max_nodes=50000;
  nan_hint_limits.ordering_hint=[](const Board&,const std::vector<Move>& moves){return std::vector<float>(moves.size(),std::numeric_limits<float>::quiet_NaN());};
  auto nan_hint=nan_hint_engine.run(start,nan_hint_limits,stop);
  assert_legal(start,nan_hint,"non-finite RRN hint");

  const std::string rrn_file="hybrid-rrn-test.weights", bad_rrn_file="hybrid-rrn-malformed.weights";
  auto rrn_weights=RrnWeights::deterministic_reference(2026); rrn_weights.save(rrn_file);
  auto restored=RrnWeights::load(rrn_file); RrnRanker roundtrip(restored);
  require(!roundtrip.learned()&&roundtrip.provenance()==rrn_a.provenance(),"RRN roundtrip provenance/learned state");
  require(roundtrip.rank(start,start_legal)==scores_a,"RRN weights roundtrip changed deterministic inference");
  std::filesystem::remove(rrn_file);
  {std::ofstream bad(bad_rrn_file,std::ios::binary);bad<<"not a CHROMATIC RRN weights file";}
  bool rejected=false; try{(void)RrnWeights::load(bad_rrn_file);}catch(const std::exception&){rejected=true;}
  std::filesystem::remove(bad_rrn_file);
  require(rejected,"malformed RRN weights file was accepted");
  std::cout<<"hybrid board/PVS regression tests passed"<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}











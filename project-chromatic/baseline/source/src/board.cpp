#include "zero/board.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace zero {
namespace {
const std::string PIECES=" PNBRQK";
std::string sq(int s) { return {char('a'+s%8),char('1'+s/8)}; }
int parse_sq(const std::string& s) {
    if(s.size()!=2 || s[0]<'a'||s[0]>'h'||s[1]<'1'||s[1]>'8') throw std::runtime_error("invalid square");
    return (s[1]-'1')*8+s[0]-'a';
}
}
std::string Move::uci() const {
    auto s=sq(from)+sq(to);
    if(promo) s += char(std::tolower(PIECES.at(promo)));
    return s;
}
int action(Move m) {
    if(m.promo>=2 && m.promo<=4) return 4096+m.from*9+(m.to%8-m.from%8+1)*3+(m.promo-2);
    return m.from*64+m.to;
}
Board Board::from_fen(const std::string& fen) {
    Board b; std::istringstream in(fen); std::string layout,side,rights,eps,extra;
    if(!(in>>layout>>side>>rights>>eps>>b.halfmove>>b.fullmove) || (in>>extra)) throw std::runtime_error("expected six FEN fields");
    int rank=7,file=0,wk=0,bk=0;
    for(char c:layout) {
        if(c=='/') { if(file!=8 || rank==0) throw std::runtime_error("invalid FEN rank"); --rank;file=0;continue; }
        if(c>='1'&&c<='8') {file+=c-'0';if(file>8) throw std::runtime_error("invalid FEN width");continue;}
        int p=int(PIECES.find(char(std::toupper(static_cast<unsigned char>(c)))));
        if(p<1||p>6||file>=8) throw std::runtime_error("invalid FEN piece");
        int color=std::isupper(static_cast<unsigned char>(c))?1:-1;
        b.squares[rank*8+file++]=color*p;
        if(p==6) {if(color==1) ++wk; else ++bk;}
    }
    if(rank!=0||file!=8||wk!=1||bk!=1||b.halfmove<0||b.fullmove<1) throw std::runtime_error("invalid FEN board or clocks");
    if(side!="w"&&side!="b") throw std::runtime_error("invalid FEN turn");
    b.turn=side=="w"?1:-1;
    for(char c:rights) { if(c=='K') b.castle|=1;else if(c=='Q') b.castle|=2;else if(c=='k') b.castle|=4;else if(c=='q') b.castle|=8;else if(c!='-') throw std::runtime_error("invalid castling rights"); }
    b.ep=eps=="-"?-1:parse_sq(eps);
    if(b.ep>=0 && (b.ep/8!=(b.turn==1?5:2)||b.squares[b.ep]!=0||b.squares[b.ep-b.turn*8]!=-b.turn)) throw std::runtime_error("invalid en passant square");
    b.history.push_back(b.key());return b;
}
std::string Board::fen() const {
    std::string out;
    for(int r=7;r>=0;--r) {int empty=0;for(int f=0;f<8;++f) {int p=squares[r*8+f];if(!p){++empty;continue;}if(empty){out+=char('0'+empty);empty=0;}char c=PIECES[std::abs(p)];out+=p>0?c:char(std::tolower(c));}if(empty)out+=char('0'+empty);if(r)out+='/';}
    out+=turn==1?" w ":" b ";std::string rights;
    for(int i=0;i<4;++i)if(castle&(1<<i))rights+="KQkq"[i];
    return out+(rights.empty()?"-":rights)+" "+(ep<0?"-":sq(ep))+" "+std::to_string(halfmove)+" "+std::to_string(fullmove);
}
bool Board::attacked(int s,int by) const {
    int x=s%8,y=s/8;
    for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx) {
        if(!dx&&!dy)continue;
        int xx=x+dx,yy=y+dy,d=1;
        while(xx>=0&&xx<8&&yy>=0&&yy<8) {
            int p=squares[yy*8+xx];
            if(p) {if(p*by>0) {int t=std::abs(p);if(t==5||(t==4&&(!dx||!dy))||(t==3&&dx&&dy)||(d==1&&t==6)||(d==1&&t==1&&dx&&dy==-by))return true;}break;}
            xx+=dx;yy+=dy;++d;
        }
    }
    constexpr int jumps[8][2]={{1,2},{2,1},{-1,2},{-2,1},{1,-2},{2,-1},{-1,-2},{-2,-1}};
    for(auto& j:jumps) {int xx=x+j[0],yy=y+j[1];if(xx>=0&&xx<8&&yy>=0&&yy<8&&squares[yy*8+xx]==by*2)return true;}
    return false;
}
bool Board::check(int color) const {for(int i=0;i<64;++i)if(squares[i]==color*6)return attacked(i,-color);return true;}
std::vector<Move> Board::legal() const {
    std::vector<Move> pseudo,result; pseudo.reserve(64);result.reserve(64);
    auto add=[&](int f,int t) {if(std::abs(squares[f])==1&&(t/8==0||t/8==7)){for(int p=2;p<=5;++p)pseudo.push_back({f,t,p});}else pseudo.push_back({f,t,0});};
    constexpr int jumps[8][2]={{1,2},{2,1},{-1,2},{-2,1},{1,-2},{2,-1},{-1,-2},{-2,-1}};
    for(int from=0;from<64;++from) {
        int p=squares[from];if(p*turn<=0)continue;int type=std::abs(p),x=from%8,y=from/8;
        if(type==1) {
            int one=from+turn*8;
            if(one>=0&&one<64&&!squares[one]) {add(from,one);if(y==(turn==1?1:6)&&!squares[from+turn*16])add(from,from+turn*16);}
            for(int dx:{-1,1}) {int xx=x+dx,yy=y+turn;if(xx<0||xx>7||yy<0||yy>7)continue;int t=yy*8+xx;
                if((squares[t]*turn<0&&std::abs(squares[t])!=6)||(t==ep&&squares[t]==0&&squares[t-turn*8]==-turn))add(from,t);
            }
        } else if(type==2) {
            for(auto& j:jumps) {int xx=x+j[0],yy=y+j[1];if(xx>=0&&xx<8&&yy>=0&&yy<8) {int t=yy*8+xx;if(squares[t]*turn<=0&&std::abs(squares[t])!=6)add(from,t);}}
        } else {
            for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx) {
                if((!dx&&!dy)||(type==3&&(!dx||!dy))||(type==4&&dx&&dy))continue;
                int xx=x+dx,yy=y+dy;
                while(xx>=0&&xx<8&&yy>=0&&yy<8) {int t=yy*8+xx;if(squares[t]*turn>0||std::abs(squares[t])==6)break;add(from,t);if(squares[t]||type==6)break;xx+=dx;yy+=dy;}
            }
            if(type==6 && from==(turn==1?4:60) && !check(turn)) {
                int base=turn==1?0:56,ks=turn==1?1:4,qs=turn==1?2:8;
                if((castle&ks)&&squares[base+7]==turn*4&&!squares[base+5]&&!squares[base+6]&&!attacked(base+5,-turn)&&!attacked(base+6,-turn))add(from,base+6);
                if((castle&qs)&&squares[base]==turn*4&&!squares[base+1]&&!squares[base+2]&&!squares[base+3]&&!attacked(base+3,-turn)&&!attacked(base+2,-turn))add(from,base+2);
            }
        }
    }
    for(auto m:pseudo) {Board b=*this;b.make(m,false);if(!b.check(turn))result.push_back(m);}
    return result;
}
void Board::make(Move m,bool track) {
    int p=squares[m.from],captured=squares[m.to],old_ep=ep;
    ep=-1;++halfmove;if(std::abs(p)==1||captured)halfmove=0;
    squares[m.to]=m.promo?turn*m.promo:p;squares[m.from]=0;
    if(std::abs(p)==1) {if(m.to==old_ep&&!captured&&m.from%8!=m.to%8)squares[m.to-turn*8]=0;if(std::abs(m.to-m.from)==16)ep=(m.to+m.from)/2;}
    if(std::abs(p)==6) {
        castle &= turn==1?~3:~12;
        if(m.to-m.from==2){squares[m.from+1]=squares[m.from+3];squares[m.from+3]=0;}
        if(m.to-m.from==-2){squares[m.from-1]=squares[m.from-4];squares[m.from-4]=0;}
    }
    for(int s:{m.from,m.to}) {if(s==0)castle&=~2;if(s==7)castle&=~1;if(s==56)castle&=~8;if(s==63)castle&=~4;}
    if(turn==-1)++fullmove;turn=-turn;
    if(track) {if(halfmove==0)history.clear();history.push_back(key());}
}
void Board::play(const std::string& text) {for(auto m:legal())if(m.uci()==text){make(m);return;}throw std::runtime_error("illegal move: "+text);}
uint64_t Board::key() const {
    uint64_t h=14695981039346656037ULL;auto mix=[&](int n){h^=uint64_t(n);h*=1099511628211ULL;};
    for(int p:squares)mix(p+6);mix(turn+1);mix(castle);
    // FIDE repetition includes an EP square only if a legal EP capture exists.
    int effective=-1;
    if(ep>=0) for(int dx:{-1,1}) {int f=ep-turn*8+dx;if(f<0||f>=64||std::abs(f%8-ep%8)!=1||squares[f]!=turn)continue;Board b=*this;b.make({f,ep,0},false);if(!b.check(turn)){effective=ep;break;}}
    mix(effective+1);return h;
}
int Board::repetitions() const {return int(std::count(history.begin(),history.end(),key()));}
bool Board::insufficient() const {
    int knights=0,bishops=0,bishop_color=-1;bool mixed=false;
    for(int s=0;s<64;++s) {int p=std::abs(squares[s]);if(p==1||p==4||p==5)return false;if(p==2)++knights;if(p==3){++bishops;int c=(s/8+s%8)%2;if(bishop_color>=0&&c!=bishop_color)mixed=true;bishop_color=c;}}
    return knights+bishops<=1 || (knights==0&&!mixed);
}
bool Board::terminal(const std::vector<Move>& moves,float& value) const {
    if(moves.empty()){value=check(turn)?-1.0f:0.0f;return true;}
    if(halfmove>=100||repetitions()>=3||insufficient()){value=0;return true;}return false;
}
std::array<float,INPUTS> Board::features() const {
    std::array<float,INPUTS> x{};
    for(int s=0;s<64;++s) {int p=squares[s];if(p)x[((p>0?0:6)+std::abs(p)-1)*64+s]=1;}
    auto fill=[&](int plane,float v){std::fill(x.begin()+plane*64,x.begin()+(plane+1)*64,v);};
    fill(12,turn==1?1.0f:0.0f);for(int i=0;i<4;++i)fill(13+i,(castle&(1<<i))?1.0f:0.0f);
    if(ep>=0)x[17*64+ep]=1;
    fill(18,std::min(halfmove,100)/100.0f);fill(19,std::min(repetitions(),3)/3.0f);return x;
}
uint64_t perft(const Board& b,int depth) {if(depth<0)throw std::runtime_error("negative perft depth");if(!depth)return 1;auto moves=b.legal();if(depth==1)return moves.size();uint64_t n=0;for(auto m:moves){Board c=b;c.make(m,false);n+=perft(c,depth-1);}return n;}
}

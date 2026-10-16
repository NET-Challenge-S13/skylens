# cumulus3d 라이브 뷰어 (UX 레퍼런스)

cumulus3d 로컬 뷰어(three.js)로, 현황판 포인트 레이어·타임라인·모니터·대기 화면의 동작 기준이다 (index.html + viewer.js).
팩 형식 v4: `manifest.json` + `packs/e_NNNN.bin`, 레코드 add/move 20 B(u32 id, 3xf32 xyz, 3xu8 rgb, u8 zone|pad), remove 4 B(u32 id), 리틀엔디언.
페이드 150 ms, 이동 글라이드 700 ms, 업로드 예산 프레임당 200k점, 되돌리기는 이벤트별 undo 로그의 역연산 애니메이션.

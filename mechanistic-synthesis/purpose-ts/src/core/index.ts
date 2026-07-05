// Pure core of @buhera/purpose.
//
// These operators are stateless and operate on a ContextGraphView (or a
// step array, for the graph builders). This is the layer Graffiti drives
// directly; the Session layer (../session) is a thin stateful wrapper
// over exactly these functions.

export type {
  StepId,
  ItemId,
  Term,
  TermSet,
  MediumId,
  Step,
  TaggedItem,
  Goal,
  WeightedEdge,
  ContextGraphView,
} from "./types.js";
export { DEFAULT_MEDIUM } from "./types.js";

export type { EdgeWeight } from "./graph.js";
export {
  identityWeight,
  buildGraph,
  floor,
  floorOfEdges,
  residue,
  termAdjacency,
  goalSeeds,
} from "./graph.js";

export { seek, reach, necessary, contribution } from "./necessity.js";

export type { CarryItem, CarryOutcome } from "./knapsack.js";
export { defaultValue, carryGreedy, carryExact } from "./knapsack.js";

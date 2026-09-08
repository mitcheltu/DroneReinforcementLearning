import editableCourse from "../editor/course.schema.json";
import vehicle from "../../../shared/schemas/vehicle.schema.json";
import observation from "../../../shared/schemas/observation.schema.json";
import rules from "../../../shared/schemas/course-rules.schema.json";
import training from "../../../shared/schemas/training.schema.json";
import config from "../../../shared/schemas/config.schema.json";
import state from "../../../shared/schemas/state.schema.json";
import course from "../../../shared/schemas/course.schema.json";
import model from "../../../shared/schemas/model.schema.json";
import replay from "../../../shared/schemas/replay.schema.json";
import worker from "../../../shared/schemas/worker-message.schema.json";

export const schemas = {
  vehicle, observation, "course-rules": rules, training, config, state, course, model,
  replay, "worker-message": worker, "editable-course": editableCourse,
};
export type SchemaName = keyof typeof schemas;

